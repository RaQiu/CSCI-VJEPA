import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

_CACHE = None


def get_jepa_lazy_cache(cfg, device):
    global _CACHE
    if _CACHE is None:
        _CACHE = JepaLazyCache(cfg, device=device)
    return _CACHE


def load_jepa_batch(cfg, encoded_paths, device, write_cache=True):
    cache = get_jepa_lazy_cache(cfg, device=device)
    tokens = cache.get_batch(encoded_paths, write_cache=write_cache).to(device=device, non_blocking=True)
    if cfg.MODEL.JEPA_RELEASE_ENCODER_AFTER_BATCH:
        cache.release_encoder()
    return tokens


class JepaLazyCache:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.height = int(cfg.DATA.IMG_HEIGHT)
        self.width = int(cfg.DATA.IMG_WIDTH)
        self.device = self._resolve_device(cfg.MODEL.JEPA_DEVICE, device)
        self.dtype = torch.float16 if cfg.MODEL.JEPA_DTYPE == "float16" else torch.bfloat16
        self.cache_root = self._resolve_cache_root(cfg)
        self.vjepa_root = self._resolve_vjepa_root(cfg)
        self.checkpoint_url = cfg.MODEL.JEPA_CHECKPOINT_URL
        self.checkpoint_file = cfg.MODEL.JEPA_CHECKPOINT_FILE
        self.encoder = None
        self.cache_root.mkdir(parents=True, exist_ok=True)

    def _resolve_device(self, requested, fallback):
        if requested == "cuda" and torch.cuda.is_available():
            return torch.device(fallback)
        if str(requested).startswith("cuda") and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(requested)

    def _resolve_cache_root(self, cfg):
        if cfg.MODEL.JEPA_CACHE_DIR:
            return Path(cfg.MODEL.JEPA_CACHE_DIR).expanduser()
        return Path(cfg.DATA.ROOT).expanduser() / "jepa_lazy_cache_vjepa2_1_g"

    def _resolve_vjepa_root(self, cfg):
        if cfg.MODEL.JEPA_ROOT:
            return Path(cfg.MODEL.JEPA_ROOT).expanduser()
        return Path(__file__).resolve().parents[2] / "vjepa2"

    def get_batch(self, encoded_paths, write_cache=True):
        paths_batch = self._decode_batch(encoded_paths)
        tensors = [self.get_one(paths, write_cache=write_cache) for paths in paths_batch]
        return torch.stack(tensors, dim=0)

    def get_one(self, paths, write_cache=True):
        cache_path = self._cache_path(paths)
        if cache_path.exists():
            return self._read(cache_path)

        if not write_cache:
            return self._extract(paths).cpu()

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = Path(str(cache_path) + ".lock")
        self._acquire_lock(lock_path)
        try:
            if cache_path.exists():
                return self._read(cache_path)
            tokens = self._extract(paths)
            self._write(cache_path, paths, tokens)
            return tokens.cpu()
        finally:
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass

    def _decode_batch(self, encoded_paths):
        if isinstance(encoded_paths, str):
            encoded_paths = [encoded_paths]
        if isinstance(encoded_paths, tuple):
            encoded_paths = list(encoded_paths)
        return [str(item).splitlines() for item in encoded_paths]

    def _cache_key(self, paths):
        meta = {
            "paths": list(paths),
            "height": self.height,
            "width": self.width,
            "model": "V-JEPA2.1-G",
            "checkpoint": self.checkpoint_file,
            "normalization": "imagenet",
            "frames": len(paths),
        }
        payload = json.dumps(meta, sort_keys=True, ensure_ascii=False)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def _cache_path(self, paths):
        key = self._cache_key(paths)
        return self.cache_root / key[:2] / f"{key}.pt"

    def _read(self, path):
        obj = torch.load(path, map_location="cpu")
        tokens = obj["tokens"] if isinstance(obj, dict) and "tokens" in obj else obj
        return tokens.to(dtype=torch.float16)

    def _write(self, path, paths, tokens):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(str(path) + f".tmp.{os.getpid()}")
        obj = {
            "tokens": tokens.detach().cpu().to(torch.float16),
            "meta": {
                "paths": list(paths),
                "height": self.height,
                "width": self.width,
                "model": "V-JEPA2.1-G",
                "checkpoint": self.checkpoint_file,
                "normalization": {
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                },
                "shape": list(tokens.shape),
            },
        }
        torch.save(obj, tmp)
        os.replace(tmp, path)

    def _acquire_lock(self, path, timeout=3600, poll=0.2):
        start = time.time()
        while True:
            try:
                fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                with os.fdopen(fd, "w") as f:
                    f.write(f"pid={os.getpid()}\n")
                return
            except FileExistsError:
                if time.time() - start > timeout:
                    raise TimeoutError(f"Timed out waiting for JEPA cache lock: {path}")
                time.sleep(poll)

    def _extract(self, paths):
        encoder = self._load_encoder()
        clip = self._load_clip(paths).to(device=self.device, dtype=self.dtype, non_blocking=True)
        with torch.no_grad(), torch.autocast(device_type=self.device.type, dtype=self.dtype, enabled=self.device.type == "cuda"):
            tokens = encoder(clip)
        return tokens.squeeze(0).detach().cpu().to(torch.float16)

    def _load_clip(self, paths):
        frames = [self._load_frame(path) for path in paths]
        clip = torch.stack(frames, dim=0)
        return clip.permute(1, 0, 2, 3).unsqueeze(0).contiguous()

    def _load_frame(self, path):
        img = Image.open(path).convert("RGB").resize((self.width, self.height), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        return (tensor - IMAGENET_MEAN) / IMAGENET_STD

    def _load_encoder(self):
        if self.encoder is not None:
            return self.encoder

        if not self.vjepa_root.exists():
            raise FileNotFoundError(f"V-JEPA root not found: {self.vjepa_root}")
        os.environ.setdefault("TORCH_HOME", str(Path(self.cfg.DATA.ROOT).expanduser() / "torch_cache"))
        sys.path.insert(0, str(self.vjepa_root.resolve()))

        from app.vjepa_2_1.models import vision_transformer as vit_encoder

        encoder = vit_encoder.__dict__["vit_gigantic_xformers"](
            patch_size=16,
            img_size=(self.height, self.width),
            num_frames=64,
            tubelet_size=2,
            use_sdpa=True,
            use_SiLU=False,
            wide_SiLU=True,
            uniform_power=False,
            use_rope=True,
            img_temporal_dim_size=1,
            interpolate_rope=True,
        )

        if self._rank() == 0:
            print(f"Loading V-JEPA2.1-G encoder from {self.checkpoint_url}")
        checkpoint = torch.hub.load_state_dict_from_url(
            self.checkpoint_url,
            map_location="cpu",
            file_name=self.checkpoint_file,
        )
        state = self._clean_backbone_key(checkpoint["target_encoder"])
        encoder.load_state_dict(state, strict=True)
        del checkpoint, state

        encoder.eval().to(device=self.device, dtype=self.dtype)
        for param in encoder.parameters():
            param.requires_grad_(False)
        self.encoder = encoder
        return self.encoder

    def release_encoder(self):
        if self.encoder is None:
            return
        del self.encoder
        self.encoder = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def _clean_backbone_key(self, state_dict):
        cleaned = {}
        for key, value in state_dict.items():
            key = key.replace("module.", "")
            key = key.replace("backbone.", "")
            cleaned[key] = value
        return cleaned

    def _rank(self):
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
        return 0
