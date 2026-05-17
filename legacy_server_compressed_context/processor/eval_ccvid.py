import time
import datetime
import logging
import numpy as np
import torch
import torch.nn.functional as F
from torch import distributed as dist
from tools.eval_metrics import evaluate, evaluate_with_clothes
from torchvision.utils import save_image 
import pickle

from processor.eval_mevid import concat_all_gather
from tools.utils import save_pickle

@torch.no_grad()
def extract_vid_feature(model, dataloader, vid2clip_index, data_length, device=None, cfg=None):
    # In build_dataloader, each original test video is split into a series of equilong clips.
    # During test, we first extact features for all clips
    clip_features, clip_pids, clip_camids, clip_clothes_ids = [], torch.tensor([]), torch.tensor([]), torch.tensor([])

    for n_iter, data in enumerate(dataloader):
        if cfg.MODEL.JEPA_ENABLE and len(data) == 7:
            imgs, pids, camids, clothes_id, clothes_ids, meta, jepa_paths = data
        else:
            imgs, pids, camids, clothes_id, clothes_ids, meta = data
            jepa_paths = None
        with torch.no_grad():
            imgs = imgs.to(device)
            meta = meta.to(device)
            clothes_ids = clothes_ids.to(device)
            meta = meta.to(torch.float32)
            jepa_tokens = None
            if cfg.MODEL.JEPA_ENABLE:
                from tools.jepa_lazy_cache import load_jepa_batch
                jepa_tokens = load_jepa_batch(cfg, jepa_paths, imgs.device)
            # torch.Size([512, 3, 224, 224])
            # torch.Size([512])

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                if cfg.MODEL.CLOTH_ONLY:
                    feat = model(imgs, clothes_ids * 0, jepa_tokens=jepa_tokens)
                else:
                    if cfg.MODEL.MASK_META:
                        meta[:, 5:21] = 0
                        meta[:, 35:57] = 0
                        meta[:, 84] = 0
                        meta[:, 90] = 0
                        meta[:, 92:96] = 0
                        meta[:, 97] = 0
                        meta[:, 100] = 0
                        meta[:, 102:105] = 0
                    if cfg.TEST.TYPE == 'image_only':
                        meta = torch.zeros_like(meta)
                    feat = model(imgs, clothes_ids * 0, meta)
            
        clip_features.append(feat.cpu())
        clip_pids = torch.cat((clip_pids, pids.cpu()), dim=0)
        clip_camids = torch.cat((clip_camids, camids.cpu()), dim=0)
        clip_clothes_ids = torch.cat((clip_clothes_ids, clothes_ids.cpu()), dim=0)
    clip_features = torch.cat(clip_features, 0)

    # Gather samples from different GPUs
    clip_features, clip_pids, clip_camids, clip_clothes_ids = \
        concat_all_gather([clip_features, clip_pids, clip_camids, clip_clothes_ids], data_length)

    # Use the averaged feature of all clips split from a video as the representation of this original full-length video
    features = torch.zeros(len(vid2clip_index), clip_features.size(1)).cuda()
    clip_features = clip_features.cuda()
    pids = torch.zeros(len(vid2clip_index))
    camids = torch.zeros(len(vid2clip_index))
    clothes_ids = torch.zeros(len(vid2clip_index))
    for i, idx in enumerate(vid2clip_index):
        features[i] = clip_features[idx[0] : idx[1], :].mean(0)
        features[i] = F.normalize(features[i], p=2, dim=0)
        pids[i] = clip_pids[idx[0]]
        camids[i] = clip_camids[idx[0]]
        clothes_ids[i] = clip_clothes_ids[idx[0]]
    features = features.cpu()

    return features, pids, camids, clothes_ids

@torch.no_grad()
def extract_vid_feature_with_index(model, dataloader, vid2clip_index, data_length, device=None, cfg=None):
    # In build_dataloader, each original test video is split into a series of equilong clips.
    # During test, we first extact features for all clips
    clip_features, clip_pids, clip_camids, clip_clothes_ids = [], torch.tensor([]), torch.tensor([]), torch.tensor([])
    clip_index =  torch.tensor([])

    for n_iter, data in enumerate(dataloader):
        if cfg.MODEL.JEPA_ENABLE and len(data) == 8:
            imgs, pids, camids, clothes_id, clothes_ids, meta, index, jepa_paths = data
        else:
            imgs, pids, camids, clothes_id, clothes_ids, meta, index = data
            jepa_paths = None
        with torch.no_grad():
            imgs = imgs.to(device)
            meta = meta.to(device)
            clothes_ids = clothes_ids.to(device)
            meta = meta.to(torch.float32)
            jepa_tokens = None
            if cfg.MODEL.JEPA_ENABLE:
                from tools.jepa_lazy_cache import load_jepa_batch
                jepa_tokens = load_jepa_batch(cfg, jepa_paths, imgs.device)
            # torch.Size([512, 3, 224, 224])
            # torch.Size([512])

            if cfg.MODEL.CLOTH_ONLY:
                feat = model(imgs, clothes_ids, jepa_tokens=jepa_tokens)
            else:
                if cfg.MODEL.MASK_META:
                    meta[:, 5:21] = 0
                    meta[:, 35:57] = 0
                    meta[:, 84] = 0
                    meta[:, 90] = 0
                    meta[:, 92:96] = 0
                    meta[:, 97] = 0
                    meta[:, 100] = 0
                    meta[:, 102:105] = 0
                if cfg.TEST.TYPE == 'image_only':
                    meta = torch.zeros_like(meta)
                feat = model(imgs, clothes_ids * 0, meta)
        
        clip_features.append(feat.cpu())
        clip_pids = torch.cat((clip_pids, pids.cpu()), dim=0)
        clip_camids = torch.cat((clip_camids, camids.cpu()), dim=0)
        clip_clothes_ids = torch.cat((clip_clothes_ids, clothes_ids.cpu()), dim=0)
        clip_index = torch.cat((clip_index, index.cpu()), dim=0)

    clip_features = torch.cat(clip_features, 0)

    # Gather samples from different GPUs
    clip_features, clip_pids, clip_camids, clip_clothes_ids, clip_index = \
        concat_all_gather([clip_features, clip_pids, clip_camids, clip_clothes_ids, clip_index], data_length)

    # Use the averaged feature of all clips split from a video as the representation of this original full-length video    
    clip_features = clip_features.cpu()
    clip_features = F.normalize(clip_features, p=2, dim=-1)
    return clip_features, clip_pids, clip_camids, clip_clothes_ids, clip_index



def compute_distance(qf, gf, output):
    # Compute distance matrix between query and gallery
    since = time.time()
    m, n = qf.size(0), gf.size(0)
    distmat = torch.zeros((m,n))
    qf, gf = qf.cuda(), gf.cuda()
    # Cosine similarity
    for i in range(m):
        distmat[i] = (- torch.mm(qf[i:i+1], gf.t())).cpu()
    distmat = distmat.numpy()
    
    time_elapsed = time.time() - since
    output('Distance computing in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    
    return distmat, qf, gf



def compute_scores(qf, gf, time_elapsed, 
        q_pids, q_camids, q_clothes_ids,  g_pids, g_camids, g_clothes_ids, 
        output=None, dataset_name=None):
    
    if dist.get_rank() == 0:
        output("Extracted features for query set, obtained {} matrix".format(qf.shape))    
        output("Extracted features for gallery set, obtained {} matrix".format(gf.shape))
        output('Extracting features complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        
    distmat, qf, gf = compute_distance(qf, gf, output)
    if dist.get_rank() == 0:
        output( f"Mean Distance : {distmat.shape}, {distmat.mean(-1)[:10]} , Mean Query : {qf.mean(-1)[:10]} Mean Gallery: {qf.mean(-1)[:10]}" )    


    q_pids, q_camids, q_clothes_ids = q_pids.numpy(), q_camids.numpy(), q_clothes_ids.numpy()
    g_pids, g_camids, g_clothes_ids = g_pids.numpy(), g_camids.numpy(), g_clothes_ids.numpy()

    since = time.time()
    if dist.get_rank() == 0:
        output("Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)
    if dist.get_rank() == 0:
        output("Results ---------------------------------------------------")
        output('top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
        output("-----------------------------------------------------------")
        time_elapsed = time.time() - since
        output('Using {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    if dist.get_rank() == 0:
        output("Computing CMC and mAP only for the same clothes setting")
    cmc, mAP = evaluate_with_clothes(distmat, q_pids, g_pids, q_camids, g_camids, q_clothes_ids, g_clothes_ids, mode='SC')
    
    if dist.get_rank() == 0:
        output("Results ---------------------------------------------------")
        output('top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
        output("-----------------------------------------------------------")

        output("Computing CMC and mAP only for clothes-changing")
    cmc, mAP = evaluate_with_clothes(distmat, q_pids, g_pids, q_camids, g_camids, q_clothes_ids, g_clothes_ids, mode='CC')
    
    if dist.get_rank() == 0:
        output("Results ---------------------------------------------------")
        output('top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
        output("-----------------------------------------------------------")
    return cmc, mAP



def test(config, model, queryloader, galleryloader, dataset, device=None, dump_w_index=None,  dump=None):
    logger = logging.getLogger('EVA-attribure')
    since = time.time()
    model.eval()
    local_rank = dist.get_rank()
    torch.cuda.synchronize()
    # Extract features 

    if dump_w_index:
        qf, q_pids, q_camids, q_clothes_ids, q_index = extract_vid_feature_with_index(model, queryloader, dataset.query_vid2clip_index,
                                    len(dataset.recombined_query), device=device, cfg=config)
        gf, g_pids, g_camids, g_clothes_ids, g_index = extract_vid_feature_with_index(model, galleryloader,  dataset.gallery_vid2clip_index,
                                    len(dataset.recombined_gallery), device=device, cfg=config)
    else:
        qf, q_pids, q_camids, q_clothes_ids = extract_vid_feature(model, queryloader, dataset.query_vid2clip_index,
                                    len(dataset.recombined_query), device=device, cfg=config)
        gf, g_pids, g_camids, g_clothes_ids = extract_vid_feature(model, galleryloader,  dataset.gallery_vid2clip_index,
                                    len(dataset.recombined_gallery), device=device, cfg=config)

    torch.cuda.empty_cache()
    time_elapsed = time.time() - since
    cmc, mAP = compute_scores(qf, gf, time_elapsed, q_pids, q_camids, q_clothes_ids,  g_pids, g_camids, g_clothes_ids, 
        output=logger.info, dataset_name=config.DATA.DATASET )
    
    if dist.get_rank() == 0:
        if dump_w_index:
            dump_pickle = dict(
                qf=qf, q_pids=q_pids, q_camids=q_camids, q_clothes_ids=q_clothes_ids, q_index=q_index,
                gf =gf, g_pids=g_pids, g_camids=g_camids, g_clothes_ids=g_clothes_ids, g_index=g_index,
            )
            save_pickle(dump_pickle, config.TAG)
        elif dump:
            dump_pickle = dict(
                qf=qf, q_pids=q_pids, q_camids=q_camids, q_clothes_ids=q_clothes_ids, 
                gf =gf, g_pids=g_pids, g_camids=g_camids, g_clothes_ids=g_clothes_ids,
            )
            save_pickle(dump_pickle, config.TAG)
        
    dist.barrier()

    return cmc[0], mAP
