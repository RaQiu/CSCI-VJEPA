# EVA models Copyright (c) 2022 BAAI-Vision
# EVA02 models Copyright (c) 2023 BAAI-Vision

import math
from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from timm.layers import PatchEmbed, Mlp, GluMlp, SwiGLU, LayerNorm, DropPath, PatchDropout, \
    apply_rot_embed_cat, apply_keep_indices_nlc, trunc_normal_, resample_patch_embed, resample_abs_pos_embed, \
    to_2tuple, use_fused_attn

from timm.models.helpers  import build_model_with_cfg
from timm.models.registry import generate_default_cfgs, register_model

from torchinfo import summary
from einops import rearrange, repeat

from typing import List, Tuple, Optional, Union
from mmengine.model import BaseModule
from timm.layers.pos_embed_sincos import build_fourier_pos_embed


######################## Untouched ########################
class EvaAttention(nn.Module):
    fused_attn: torch.jit.Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = True,
            qkv_fused: bool = True,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            attn_head_dim: Optional[int] = None,
            norm_layer: Optional[Callable] = None,
        ):
        """

        Args:
            dim:
            num_heads:
            qkv_bias:
            qkv_fused:
            attn_drop:
            proj_drop:
            attn_head_dim:
            norm_layer:
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        if qkv_fused:
            self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
            self.q_proj = self.k_proj = self.v_proj = None
            if qkv_bias:
                self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
                self.register_buffer('k_bias', torch.zeros(all_head_dim), persistent=False)
                self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
            else:
                self.q_bias = self.k_bias = self.v_bias = None
        else:
            self.q_proj = nn.Linear(dim, all_head_dim, bias=qkv_bias)
            self.k_proj = nn.Linear(dim, all_head_dim, bias=False)
            self.v_proj = nn.Linear(dim, all_head_dim, bias=qkv_bias)
            self.qkv = None
            self.q_bias = self.k_bias = self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.norm = norm_layer(all_head_dim) if norm_layer is not None else nn.Identity()
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
            self,
            x,
            rope: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
        ):
        B, N, C = x.shape

        if self.qkv is not None:
            qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
            qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
            qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
        else:
            q = self.q_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)  # B, num_heads, N, C
            k = self.k_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)
            v = self.v_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)

        # k.shape      torch.Size([8, 16, 257, 64]), 
        # rope.shape   torch.Size([256, 128])
        # apply_rot_embed_cat --> torch.Size([8, 16, 256, 64])
        if rope is not None:
            q = torch.cat([q[:, :, :1, :], apply_rot_embed_cat(q[:, :, 1:, :], rope)], 2).type_as(v)
            k = torch.cat([k[:, :, :1, :], apply_rot_embed_cat(k[:, :, 1:, :], rope)], 2).type_as(v)

        
        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p,
            )
        else:
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))
            attn = attn.softmax(dim=-1)
            if attn_mask is not None:
                attn_mask = attn_mask.to(torch.bool)
                attn = attn.masked_fill(~attn_mask[:, None, None, :], float("-inf"))
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.norm(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class EvaBlock(nn.Module):

    def __init__(
            self,
            dim: int,
            num_heads: int,
            qkv_bias: bool = True,
            qkv_fused: bool = True,
            mlp_ratio: float = 4.,
            swiglu_mlp: bool = False,
            scale_mlp: bool = False,
            scale_attn_inner: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: float = 0.,
            init_values: Optional[float] = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            attn_head_dim: Optional[int] = None,
        ):
        """

        Args:
            dim:
            num_heads:
            qkv_bias:
            qkv_fused:
            mlp_ratio:
            swiglu_mlp:
            scale_mlp:
            scale_attn_inner:
            proj_drop:
            attn_drop:
            drop_path:
            init_values:
            act_layer:
            norm_layer:
            attn_head_dim:
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = EvaAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qkv_fused=qkv_fused,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            attn_head_dim=attn_head_dim,
            norm_layer=norm_layer if scale_attn_inner else None,
        )
        self.gamma_1 = nn.Parameter(init_values * torch.ones(dim)) if init_values is not None else None
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        hidden_features = int(dim * mlp_ratio)
        if swiglu_mlp:
            if scale_mlp:
                # when norm in SwiGLU used, an impl with separate fc for gate & x is used
                self.mlp = SwiGLU(
                    in_features=dim,
                    hidden_features=hidden_features,
                    norm_layer=norm_layer if scale_mlp else None,
                    drop=proj_drop,
                )
            else:
                # w/o any extra norm, an impl with packed weights is used, matches existing GluMLP
                self.mlp = GluMlp(
                    in_features=dim,
                    hidden_features=hidden_features * 2,
                    norm_layer=norm_layer if scale_mlp else None,
                    act_layer=nn.SiLU,
                    gate_last=False,
                    drop=proj_drop,
                )
        else:
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=hidden_features,
                act_layer=act_layer,
                norm_layer=norm_layer if scale_mlp else None,
                drop=proj_drop,
            )
        self.gamma_2 = nn.Parameter(init_values * torch.ones(dim)) if init_values is not None else None
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, rope: Optional[torch.Tensor] = None, attn_mask: Optional[torch.Tensor] = None):
        if self.gamma_1 is None:
            x = x + self.drop_path1(self.attn(self.norm1(x), rope=rope, attn_mask=attn_mask))
            x = x + self.drop_path2(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), rope=rope, attn_mask=attn_mask))
            x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
        return x

class EvaBlockPostNorm(nn.Module):
    """ EVA block w/ post-norm and support for swiglu, MLP norm scale, ROPE. """
    def __init__(
            self,
            dim: int,
            num_heads: int,
            qkv_bias: bool = True,
            qkv_fused: bool = True,
            mlp_ratio: float = 4.,
            swiglu_mlp: bool = False,
            scale_mlp: bool = False,
            scale_attn_inner: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: float = 0.,
            init_values: Optional[float] = None,  # ignore for post-norm
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = nn.LayerNorm,
            attn_head_dim: Optional[int] = None,
        ):
        """

        Args:
            dim:
            num_heads:
            qkv_bias:
            qkv_fused:
            mlp_ratio:
            swiglu_mlp:
            scale_mlp:
            scale_attn_inner:
            proj_drop:
            attn_drop:
            drop_path:
            init_values:
            act_layer:
            norm_layer:
            attn_head_dim:
        """
        super().__init__()
        self.attn = EvaAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qkv_fused=qkv_fused,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            attn_head_dim=attn_head_dim,
            norm_layer=norm_layer if scale_attn_inner else None,
        )
        self.norm1 = norm_layer(dim)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        hidden_features = int(dim * mlp_ratio)
        if swiglu_mlp:
            if scale_mlp:
                # when norm in SwiGLU used, an impl with separate fc for gate & x is used
                self.mlp = SwiGLU(
                    in_features=dim,
                    hidden_features=hidden_features,
                    norm_layer=norm_layer if scale_mlp else None,
                    drop=proj_drop,
                )
            else:
                # w/o any extra norm, an impl with packed fc1 weights is used, matches existing GluMLP
                self.mlp = GluMlp(
                    in_features=dim,
                    hidden_features=hidden_features * 2,
                    norm_layer=norm_layer if scale_mlp else None,
                    act_layer=nn.SiLU,
                    gate_last=False,
                    drop=proj_drop,
                )
        else:
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=hidden_features,
                act_layer=act_layer,
                norm_layer=norm_layer if scale_mlp else None,
                drop=proj_drop,
            )
        self.norm2 = norm_layer(dim)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, rope: Optional[torch.Tensor] = None, attn_mask: Optional[torch.Tensor] = None):
        x = x + self.drop_path1(self.norm1(self.attn(x, rope=rope, attn_mask=attn_mask)))
        x = x + self.drop_path2(self.norm2(self.mlp(x)))
        return x

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': OPENAI_CLIP_MEAN, 'std': OPENAI_CLIP_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        'license': 'mit', **kwargs
    }

default_cfgs = generate_default_cfgs({
    'eva02_large_patch14_224.mim_in22k': _cfg(
        #hf_hub_id='Yuxin-CV/EVA-02', hf_hub_filename='eva02/pt/eva02_L_pt_in21k_p14.pt',
        hf_hub_id='timm/',
        num_classes=0,
    ),
    'eva02_large_patch14_224.mim_m38m': _cfg(
        #hf_hub_id='Yuxin-CV/EVA-02', hf_hub_filename='eva02/pt/eva02_L_pt_m38m_p14.pt',
        hf_hub_id='timm/',
        num_classes=0,
    ),
    'eva02_large_patch14_clip_224.merged2b': _cfg(
        # hf_hub_id='QuanSun/EVA-CLIP', hf_hub_filename='EVA02_CLIP_L_psz14_s4B.pt',
        hf_hub_id='timm/eva02_large_patch14_clip_224.merged2b_s4b_b131k',  # float16 weights
        hf_hub_filename='open_clip_pytorch_model.bin',
        num_classes=768,
    ),
})



######################## New Code ########################
class PatchEmbed3D(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_chans=3,
            embed_dim=768,
            norm_layer=None,
            flatten=True,
            bias=True,
            time_kernel=2, 
            temporal_dim=None,
        ):
        super().__init__()

        self.time_kernel = time_kernel
        self.temporal_dim = temporal_dim

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patch_size = (time_kernel, patch_size[0], patch_size[1] )
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (temporal_dim // time_kernel, img_size[0] // patch_size[1], img_size[1] // patch_size[2])
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        self.flatten = flatten

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, T, H, W = x.shape
        assert T == self.temporal_dim # T // 2 == 2 (self.grid_size --> positional embeddings)

        x = self.proj(x)
        if self.flatten:
            x = rearrange(x, "B C T H W -> B (T H W) C") # BCHW -> BNC
        x = self.norm(x)
        return x

def resample_abs_temporal_pos_embed(
        posemb,
        new_size: List[int],
        old_size: Optional[List[int]] = None,
        num_prefix_tokens: int = 1,
        interpolation: str = 'trilinear',
        antialias: bool = True,
        verbose: bool = False,
        time_stamp=None, 
    ):        
    new_size = to_2tuple(new_size)
    new_ntok = new_size[0] * new_size[1] * new_size[2]
    if not old_size:
        old_size = int(math.sqrt(posemb.shape[1] - num_prefix_tokens))
    old_size = to_2tuple(old_size)
    old_size = (1, old_size[0], old_size[1])
    if new_size == old_size:  # might not both be same container type
        return posemb

    if num_prefix_tokens:
        posemb_prefix, posemb = posemb[:, :num_prefix_tokens], posemb[:, num_prefix_tokens:]
    else:
        posemb_prefix, posemb = None, posemb

    # do the interpolation
    posemb = posemb.reshape(1, old_size[0], old_size[1], old_size[2], -1).permute(0, 4, 1, 2, 3)
    # posemb = F.interpolate(posemb, size=new_size, mode=interpolation, antialias=antialias)
    posemb = F.interpolate(posemb.float(), size=new_size, mode=interpolation)
    posemb = posemb.permute(0, 2, 3, 4, 1).reshape(1, new_ntok, -1)

    # add back extra (class, etc) prefix tokens
    if posemb_prefix is not None:
        # print(posemb_prefix.shape, posemb.shape)
        posemb = torch.cat([posemb_prefix, posemb], dim=1)
    return posemb

def build_rotary_pos_embed(
        feat_shape: List[int],
        bands: Optional[torch.Tensor] = None,
        dim: int = 64,
        max_res: int = 224,
        temperature: float = 10000.,
        linear_bands: bool = False,
        in_pixels: bool = True,
        ref_feat_shape: Optional[List[int]] = None,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ):
    sin_emb, cos_emb = build_fourier_pos_embed(
        feat_shape,
        bands=bands,
        num_bands=dim // 4,
        max_res=max_res,
        temperature=temperature,
        linear_bands=linear_bands,
        in_pixels=in_pixels,
        ref_feat_shape=ref_feat_shape,
        device=device,
        dtype=dtype,
    )
    
    new_size = sin_emb.size()
    T,H,W, D_old, C = sin_emb.shape
    sin_emb = rearrange(sin_emb, " T H W N C -> T C N H W")
    cos_emb = rearrange(cos_emb, " T H W N C -> T C N H W")

    sin_emb = F.interpolate(sin_emb.float(), size=(2, H, W), mode="trilinear")
    cos_emb = F.interpolate(cos_emb.float(), size=(2, H, W), mode="trilinear")

    sin_emb = rearrange(sin_emb, " T C N H W -> T H W N C")
    cos_emb = rearrange(cos_emb, " T C N H W -> T H W N C")

    # sin_emb :: (H, W, 2, num_bands) --> (T, H, W, 3, num_bands)  
    num_spatial_dim = 1
    # this would be much nicer as a .numel() call to torch.Size(), but torchscript sucks
    for x in feat_shape:
        num_spatial_dim *= x
    sin_emb = sin_emb.reshape(num_spatial_dim, -1).repeat_interleave(2, -1)
    cos_emb = cos_emb.reshape(num_spatial_dim, -1).repeat_interleave(2, -1)
    return sin_emb, cos_emb

class RotaryEmbeddingCat_Temporal(nn.Module):
    """ Rotary position embedding w/ concatenatd sin & cos

    The following impl/resources were referenced for this impl:
    * https://github.com/lucidrains/vit-pytorch/blob/6f3a5fcf0bca1c5ec33a35ef48d97213709df4ba/vit_pytorch/rvt.py
    * https://blog.eleuther.ai/rotary-embeddings/
    """

    def __init__(
            self,
            dim,
            max_res=224,
            temperature=10000,
            in_pixels=True,
            linear_bands: bool = False,
            feat_shape: Optional[List[int]] = None,
            ref_feat_shape: Optional[List[int]] = None,
    ):
        super().__init__()
        self.dim = dim
        self.max_res = max_res
        self.temperature = temperature
        self.in_pixels = in_pixels
        self.feat_shape = feat_shape
        self.ref_feat_shape = ref_feat_shape

        if feat_shape is None:
            # only cache bands
            if in_pixels:
                bands = pixel_freq_bands(
                    dim // 4,
                    float(max_res),
                    linear_bands=linear_bands,
                )
            else:
                bands = freq_bands(
                    dim // 4,
                    temperature=temperature,
                    step=1,
                )
            self.register_buffer(
                'bands',
                bands,
                persistent=False,
            )
            self.pos_embed = None
        else:
            # cache full sin/cos embeddings if shape provided up front
            embeds = build_rotary_pos_embed( feat_shape=feat_shape, dim=dim, max_res=max_res, linear_bands=linear_bands, in_pixels=in_pixels, ref_feat_shape=self.ref_feat_shape, )
            self.bands = None
            self.register_buffer(
                'pos_embed',
                torch.cat(embeds, -1),
                persistent=False,
            )

    def get_embed(self, shape: Optional[List[int]] = None):
        if self.bands is not None and shape is not None:
            # rebuild embeddings every call, use if target shape changes
            embeds = build_rotary_pos_embed( shape, self.bands, in_pixels=self.in_pixels, ref_feat_shape=self.ref_feat_shape, )
            return torch.cat(embeds, -1)
        elif self.pos_embed is not None:
            return self.pos_embed
        else:
            assert False, "get_embed() requires pre-computed pos_embed or valid shape w/ pre-computed bands"

    def forward(self, x):
        # assuming channel-first tensor where spatial dim are >= 2
        pos_embed = self.get_embed(x.shape[2:])
        return apply_rot_embed_cat(x, pos_embed)

class Eva_Brute_3D(nn.Module):
    def __init__(
            self,
            img_size: Union[int, Tuple[int, int]] = 224,
            patch_size: Union[int, Tuple[int, int]] = 16,
            in_chans: int = 3,
            num_classes: int = 1000,
            global_pool: str = 'avg',
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            qkv_bias: bool = True,
            qkv_fused: bool = True,
            mlp_ratio: float = 4.,
            swiglu_mlp: bool = False,
            scale_mlp: bool = False,
            scale_attn_inner: bool = False,
            drop_rate: float = 0.,
            pos_drop_rate: float = 0.,
            patch_drop_rate: float = 0.,
            proj_drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            norm_layer: Callable = LayerNorm,
            init_values: Optional[float] = None,
            class_token: bool = True,
            use_abs_pos_emb: bool = True,
            use_rot_pos_emb: bool = False,
            use_post_norm: bool = False,
            ref_feat_shape: Optional[Union[Tuple[int, int], int]] = None,
            head_init_scale: float = 0.001,
            cloth: int = 0,
            cloth_xishu: int = 3,
            temporal_dim=4,
        ):
        
        super().__init__()
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_prefix_tokens = 1 if class_token else 0
        self.grad_checkpointing = False

        
        self.patch_embed = PatchEmbed3D(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            temporal_dim=temporal_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        # self.cloth_xishu = cloth_xishu
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + self.num_prefix_tokens, embed_dim)) if use_abs_pos_emb else None
        # self.cloth_embed = nn.Parameter(torch.zeros(cloth, 1, embed_dim))
        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        if patch_drop_rate > 0:
            self.patch_drop = PatchDropout(
                patch_drop_rate,
                num_prefix_tokens=self.num_prefix_tokens,
                return_indices=True,
            )
        else:
            self.patch_drop = None

        if use_rot_pos_emb:
            ref_feat_shape = to_2tuple(ref_feat_shape) if ref_feat_shape is not None else None
            ref_feat_shape = ( self.patch_embed.grid_size[0], ref_feat_shape[0], ref_feat_shape[1])
            self.rope = RotaryEmbeddingCat_Temporal( embed_dim // num_heads, in_pixels=False, feat_shape=self.patch_embed.grid_size, ref_feat_shape=ref_feat_shape, )
        else:
            self.rope = None

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        block_fn = EvaBlockPostNorm if use_post_norm else EvaBlock
        self.blocks = nn.ModuleList([
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qkv_fused=qkv_fused,
                mlp_ratio=mlp_ratio,
                swiglu_mlp=swiglu_mlp,
                scale_mlp=scale_mlp,
                scale_attn_inner=scale_attn_inner,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                init_values=init_values,
            )
            for i in range(depth)])

        use_fc_norm = self.global_pool == 'avg'
        self.norm = nn.Identity() if use_fc_norm else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)
        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)
        # trunc_normal_(self.cloth_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)

        self.fix_init_weight()
        if isinstance(self.head, nn.Linear):
            trunc_normal_(self.head.weight, std=.02)
            self.head.weight.data.mul_(head_init_scale)
            self.head.bias.data.mul_(head_init_scale)

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    @torch.jit.ignore
    def no_weight_decay(self):
        nwd = {'pos_embed', 'cls_token'}
        return nwd

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(
            stem=r'^cls_token|pos_embed|patch_embed',  # stem and embed
            blocks=[(r'^blocks\.(\d+)', None), (r'^norm', (99999,))],
        )
        return matcher

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            self.global_pool = global_pool
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x,cloth_id):
        x = self.patch_embed(x)

        if self.cls_token is not None:
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)

        # apply abs position embedding
        if self.pos_embed is not None:
            # if self.training:
            # x = x + self.pos_embed + self.cloth_xishu * self.cloth_embed[cloth_id]
            x = x + self.pos_embed
        x = self.pos_drop(x)

        # obtain shared rotary position embedding and apply patch dropout
        rot_pos_embed = self.rope.get_embed() if self.rope is not None else None
        if self.patch_drop is not None:
            x, keep_indices = self.patch_drop(x)
            if rot_pos_embed is not None and keep_indices is not None:
                rot_pos_embed = apply_keep_indices_nlc(x, rot_pos_embed, keep_indices)

        for blk in self.blocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(blk, x, rope=rot_pos_embed)
            else:
                x = blk(x, rope=rot_pos_embed)

        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool:
            x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x,cloth_id):
        x = x[:, : , ::2]
        
        x = self.forward_features(x,cloth_id)
        feat = self.forward_head(x, pre_logits=True)
        if not self.training:
            return feat
        else:
            cls_score = self.head(feat)
            return cls_score, feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))
    
def checkpoint_filter_fn(
        state_dict,
        model,
        interpolation='bicubic',
        antialias=True,
    ):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    state_dict = state_dict.get('model_ema', state_dict)
    state_dict = state_dict.get('model', state_dict)
    state_dict = state_dict.get('module', state_dict)
    state_dict = state_dict.get('state_dict', state_dict)
    # prefix for loading OpenCLIP compatible weights
    if 'visual.trunk.pos_embed' in state_dict:
        prefix = 'visual.trunk.'
    elif 'visual.pos_embed' in state_dict:
        prefix = 'visual.'
    else:
        prefix = ''
    mim_weights = prefix + 'mask_token' in state_dict
    no_qkv = prefix + 'blocks.0.attn.q_proj.weight' in state_dict

    len_prefix = len(prefix)
    for k, v in state_dict.items():
        if prefix:
            if k.startswith(prefix):
                k = k[len_prefix:]
            else:
                continue

        if 'rope' in k:
            # fixed embedding no need to load buffer from checkpoint
            continue

        if 'patch_embed.proj.weight' in k:
            _, _, T, H, W = model.patch_embed.proj.weight.shape
            # if v.shape[-1] != W or v.shape[-2] != H:
            v = v.unsqueeze(2).repeat(1,1,T,1,1) / T
        elif k == 'pos_embed' and v.shape[1] != model.pos_embed.shape[1]:
            # To resize pos embedding when using model at different size from pretrained weights
            num_prefix_tokens = 0 if getattr(model, 'no_embed_class', False) else getattr(model, 'num_prefix_tokens', 1)
            v = resample_abs_temporal_pos_embed(
                v,
                new_size=model.patch_embed.grid_size,
                num_prefix_tokens=num_prefix_tokens,
                antialias=antialias,
                verbose=True,
            )

        k = k.replace('mlp.ffn_ln', 'mlp.norm')
        k = k.replace('attn.inner_attn_ln', 'attn.norm')
        k = k.replace('mlp.w12', 'mlp.fc1')
        k = k.replace('mlp.w1', 'mlp.fc1_g')
        k = k.replace('mlp.w2', 'mlp.fc1_x')
        k = k.replace('mlp.w3', 'mlp.fc2')
        if no_qkv:
            k = k.replace('q_bias', 'q_proj.bias')
            k = k.replace('v_bias', 'v_proj.bias')

        if mim_weights and k in ('mask_token', 'lm_head.weight', 'lm_head.bias', 'norm.weight', 'norm.bias'):
            if k == 'norm.weight' or k == 'norm.bias':
                # try moving norm -> fc norm on fine-tune, probably a better starting point than new init
                k = k.replace('norm', 'fc_norm')
            else:
                # skip pretrain mask token & head weights
                continue

        out_dict[k] = v

    return out_dict

def _create_eva(variant, pretrained=False, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Eva models.')

    model = build_model_with_cfg(
        Eva_Brute_3D, variant, pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        **kwargs)
    return model


@register_model
def eva02_brute_3D(pretrained=False, **kwargs) :
    """ A EVA-CLIP specific variant that adds additional attn scale layernorm to eva02_large """
    model_args = dict(
        img_size=224,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4 * 2 / 3,
        qkv_fused=False,
        swiglu_mlp=True,
        scale_mlp=True,
        scale_attn_inner=True,
        use_rot_pos_emb=True,
        ref_feat_shape=(16, 16),  # 224/14
        global_pool=kwargs.pop('global_pool', 'token'),
        cloth=300,
    )
    model = _create_eva('eva02_large_patch14_clip_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model







if __name__ == "__main__":
    x = torch.randn([2, 3, 4, 224, 224])
    cloth_id = torch.tensor([2, 3])
    model = eva02_brute_3D(pretrained=True)
    output = model(x, cloth_id)
    print(output[0].shape)
    # print(summary(model, input_size=((8, 3, 224, 224),(3,7))))
    
# cd ~/MADE_ReID/
# python model/eva_cloth_vid.py
