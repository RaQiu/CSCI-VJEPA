import time
import datetime
import logging
import numpy as np
import torch
import torch.nn.functional as F
from torch import distributed as dist
# from torchvision.utils import save_image 
import pickle
import os
import os.path as osp
from einops import rearrange, repeat
from tools.utils import save_pickle
from tools.utils import save_image, normalize

def compute_ap_cmc(index, good_index, junk_index):
    """ Compute AP and CMC for each sample
    """
    ap = 0
    cmc = np.zeros(len(index)) 
    
    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask==True)
    rows_good = rows_good.flatten()
    
    cmc[rows_good[0]:] = 1.0
    for i in range(ngood):
        d_recall = 1.0/ngood
        precision = (i+1)*1.0/(rows_good[i]+1)
        ap = ap + d_recall*precision

    return ap, cmc

def evaluate(distmat, q_pids, g_pids, q_camids, g_camids):
    """ Compute CMC and mAP

    Args:
        distmat (numpy ndarray): distance matrix with shape (num_query, num_gallery).
        q_pids (numpy array): person IDs for query samples.
        g_pids (numpy array): person IDs for gallery samples.
        q_camids (numpy array): camera IDs for query samples.
        g_camids (numpy array): camera IDs for gallery samples.
    """
    num_q, num_g = distmat.shape
    index = np.argsort(distmat, axis=1) # from small to large

    num_no_gt = 0 # num of query imgs without groundtruth
    num_r1 = 0
    CMC = np.zeros(len(g_pids))
    AP = 0

    for i in range(num_q):
        # groundtruth index
        query_index = np.argwhere(g_pids==q_pids[i])
        camera_index = np.argwhere(g_camids==q_camids[i])
        good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
        if good_index.size == 0:
            num_no_gt += 1
            continue
        # remove gallery samples that have the same pid and camid with query
        junk_index = np.intersect1d(query_index, camera_index)

        ap_tmp, CMC_tmp = compute_ap_cmc(index[i], good_index, junk_index)
        if CMC_tmp[0]==1:
            num_r1 += 1
        CMC = CMC + CMC_tmp
        AP += ap_tmp

    if num_no_gt > 0:
        logger = logging.getLogger('reid.evaluate')
        logger.info("{} query samples do not have groundtruth.".format(num_no_gt))

    CMC = CMC / (num_q - num_no_gt)
    mAP = AP / (num_q - num_no_gt)

    return CMC, mAP

def evaluate_with_clothes(distmat, q_pids, g_pids, q_camids, g_camids, q_clothids, g_clothids, mode='CC', output=None):
    """ Compute CMC and mAP with clothes

    Args:
        distmat (numpy ndarray): distance matrix with shape (num_query, num_gallery).
        q_pids (numpy array): person IDs for query samples.
        g_pids (numpy array): person IDs for gallery samples.
        q_camids (numpy array): camera IDs for query samples.
        g_camids (numpy array): camera IDs for gallery samples.
        q_clothids (numpy array): clothes IDs for query samples.
        g_clothids (numpy array): clothes IDs for gallery samples.
        mode: 'CC' for clothes-changing; 'SC' for the same clothes.
    """
    assert mode in ['CC', 'SC']
    
    num_q, num_g = distmat.shape
    index = np.argsort(distmat, axis=1) # from small to large

    num_no_gt = 0 # num of query imgs without groundtruth
    num_r1 = 0
    CMC = np.zeros(len(g_pids))
    AP = 0

    for i in range(num_q):
        # groundtruth index
        query_index = np.argwhere(g_pids==q_pids[i])
        camera_index = np.argwhere(g_camids==q_camids[i])
        cloth_index = np.argwhere(g_clothids==q_clothids[i])
        good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
        if mode == 'CC':
            good_index = np.setdiff1d(good_index, cloth_index, assume_unique=True)
            # remove gallery samples that have the same (pid, camid) or (pid, clothid) with query
            junk_index1 = np.intersect1d(query_index, camera_index)
            junk_index2 = np.intersect1d(query_index, cloth_index)
            junk_index = np.union1d(junk_index1, junk_index2)
        else:
            good_index = np.intersect1d(good_index, cloth_index)
            # remove gallery samples that have the same (pid, camid) or 
            # (the same pid and different clothid) with query
            junk_index1 = np.intersect1d(query_index, camera_index)
            junk_index2 = np.setdiff1d(query_index, cloth_index)
            junk_index = np.union1d(junk_index1, junk_index2)

        if good_index.size == 0:
            num_no_gt += 1
            continue
    
        ap_tmp, CMC_tmp = compute_ap_cmc(index[i], good_index, junk_index)
        if CMC_tmp[0]==1:
            num_r1 += 1
        CMC = CMC + CMC_tmp
        AP += ap_tmp

    if num_no_gt > 0:
        output("{} query samples do not have groundtruth.".format(num_no_gt))

    if (num_q - num_no_gt) != 0:
        CMC = CMC / (num_q - num_no_gt)
        mAP = AP / (num_q - num_no_gt)
    else:
        mAP = 0

    return CMC, mAP

def concat_all_gather(tensors, num_total_examples):
    '''
    Performs all_gather operation on the provided tensor list.
    '''
    outputs = []
    for tensor in tensors:
        tensor = tensor.cuda()
        tensors_gather = [tensor.clone() for _ in range(dist.get_world_size())]
        dist.all_gather(tensors_gather, tensor)
        output = torch.cat(tensors_gather, dim=0).cpu()
        # truncate the dummy elements added by DistributedInferenceSampler
        outputs.append(output[:num_total_examples])
    return outputs



def evaluate_with_locations(distmat, q_pids, g_pids, q_camids, g_camids, mode='SL', config=None, output=None):
    """ Compute CMC and mAP with locations

    Args:
        distmat (numpy ndarray): distance matrix with shape (num_query, num_gallery).
        q_pids (numpy array): person IDs for query samples.
        g_pids (numpy array): person IDs for gallery samples.
        q_camids (numpy array): camera IDs for query samples.
        g_camids (numpy array): camera IDs for gallery samples.
        mode: 'SL' for same locations; 'DL' for different locations.
    """
    assert mode in ['SL', 'DL']
    
    in_cam = [330, 329, 507, 508, 509]
    out_cam = [436, 505, 336, 340, 639, 301]

    dataset_dir = config.DATA.ROOT
    # dataset_dir = '../../mevid'
    query_IDX_path = osp.join(dataset_dir, 'query_IDX.txt')
    query_IDX = np.loadtxt(query_IDX_path).astype(int)

    camera_file = osp.join(dataset_dir, 'test_track_scale.txt')
    camera_set = np.genfromtxt(camera_file,dtype='str')[:,0]
    q_locationids = camera_set[query_IDX]
    gallery_IDX = [i for i in range(camera_set.shape[0]) if i not in query_IDX]
    g_locationids = camera_set[gallery_IDX]
    for k in range(q_locationids.shape[0]):
        q_locationids[k] = 0 if int(q_locationids[k][9:12]) in in_cam else 1
    for k in range(g_locationids.shape[0]):
        g_locationids[k] = 0 if int(g_locationids[k][9:12]) in in_cam else 1

    num_q, num_g = distmat.shape
    index = np.argsort(distmat, axis=1) # from small to large

    num_no_gt = 0 # num of query imgs without groundtruth
    num_r1 = 0
    CMC = np.zeros(len(g_pids))
    AP = 0

    for i in range(num_q):
        # groundtruth index
        query_index = np.argwhere(g_pids==q_pids[i])
        camera_index = np.argwhere(g_camids==q_camids[i])
        location_index = np.argwhere(g_locationids==q_locationids[i])
        good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
        if mode == 'DL':
            good_index = np.setdiff1d(good_index, location_index, assume_unique=True)
            # remove gallery samples that have the same (pid, camid) or (pid, clothid) with query
            junk_index1 = np.intersect1d(query_index, camera_index)
            junk_index2 = np.intersect1d(query_index, location_index)
            junk_index = np.union1d(junk_index1, junk_index2)
        else:
            good_index = np.intersect1d(good_index, location_index)
            # remove gallery samples that have the same (pid, camid) or 
            # (the same pid and different clothid) with query
            junk_index1 = np.intersect1d(query_index, camera_index)
            junk_index2 = np.setdiff1d(query_index, location_index)
            junk_index = np.union1d(junk_index1, junk_index2)

        if good_index.size == 0:
            num_no_gt += 1
            continue
    
        ap_tmp, CMC_tmp = compute_ap_cmc(index[i], good_index, junk_index)
        if CMC_tmp[0]==1:
            num_r1 += 1
        CMC = CMC + CMC_tmp
        AP += ap_tmp

    if num_no_gt > 0:
        output("{} query samples do not have groundtruth.".format(num_no_gt))

    if (num_q - num_no_gt) != 0:
        CMC = CMC / (num_q - num_no_gt)
        mAP = AP / (num_q - num_no_gt)
    else:
        mAP = 0

    return CMC, mAP

def evaluate_with_scales(distmat, q_pids, g_pids, q_camids, g_camids, mode='SS', config=None, output=None):
    """ Compute CMC and mAP with scales

    Args:
        distmat (numpy ndarray): distance matrix with shape (num_query, num_gallery).
        q_pids (numpy array): person IDs for query samples.
        g_pids (numpy array): person IDs for gallery samples.
        q_camids (numpy array): camera IDs for query samples.
        g_camids (numpy array): camera IDs for gallery samples.
        mode: 'SS' for same size; 'DS' for diff. size.
    """
    assert mode in ['SS', 'DS']
    
    dataset_dir = config.DATA.ROOT
    # dataset_dir = '../../mevid'
    query_IDX_path = osp.join(dataset_dir, 'query_IDX.txt')
    query_IDX = np.loadtxt(query_IDX_path).astype(int)

    scale_file = osp.join(dataset_dir, 'test_track_scale.txt')
    # scale_file = '../test_track_scale.txt'
    scale_set = np.genfromtxt(scale_file,dtype='str')[:,1]
    q_scaleids = scale_set[query_IDX]
    gallery_IDX = [i for i in range(scale_set.shape[0]) if i not in query_IDX]
    g_scaleids = scale_set[gallery_IDX]

    num_q, num_g = distmat.shape
    index = np.argsort(distmat, axis=1) # from small to large

    num_no_gt = 0 # num of query imgs without groundtruth
    num_r1 = 0
    CMC = np.zeros(len(g_pids))
    AP = 0

    for i in range(num_q):
        # groundtruth index
        query_index = np.argwhere(g_pids==q_pids[i])
        camera_index = np.argwhere(g_camids==q_camids[i])
        scale_index = np.argwhere(g_scaleids==q_scaleids[i])
        good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
        if mode == 'DS':
            good_index = np.setdiff1d(good_index, scale_index, assume_unique=True)
            # remove gallery samples that have the same (pid, camid) or (pid, scaleid) with query
            junk_index1 = np.intersect1d(query_index, camera_index)
            junk_index2 = np.intersect1d(query_index, scale_index)
            junk_index = np.union1d(junk_index1, junk_index2)
        else:
            good_index = np.intersect1d(good_index, scale_index)
            # remove gallery samples that have the same (pid, camid) or 
            # (the same pid and different scaleid) with query
            junk_index1 = np.intersect1d(query_index, camera_index)
            junk_index2 = np.setdiff1d(query_index, scale_index)
            junk_index = np.union1d(junk_index1, junk_index2)

        if good_index.size == 0:
            num_no_gt += 1
            continue
    
        ap_tmp, CMC_tmp = compute_ap_cmc(index[i], good_index, junk_index)
        if CMC_tmp[0]==1:
            num_r1 += 1
        CMC = CMC + CMC_tmp
        AP += ap_tmp

    if num_no_gt > 0:
        output("{} query samples do not have groundtruth.".format(num_no_gt))

    if (num_q - num_no_gt) != 0:
        CMC = CMC / (num_q - num_no_gt)
        mAP = AP / (num_q - num_no_gt)
    else:
        mAP = 0

    return CMC, mAP






@torch.no_grad()
def extract_mevid_vid_feature(logger, model, dataloader, vid2clip_index, data_length, device, cfg):
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
                write_cache = cfg.MODEL.JEPA_WRITE_IMAGE_CACHE if imgs.dim() == 4 else cfg.MODEL.JEPA_WRITE_TRAIN_CACHE
                jepa_tokens = load_jepa_batch(cfg, jepa_paths, imgs.device, write_cache=write_cache)
            # torch.Size([512, 3, 224, 224])
            # torch.Size([512])

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                # save_image(normalize(rearrange(imgs[:10], "B C N ... -> (B N) C ...")), "temp-2.png")
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

    # dist.barrier()
    clip_features = torch.cat(clip_features, 0)

    # Gather samples from different GPUs
    clip_features, clip_pids, clip_camids, clip_clothes_ids = \
        concat_all_gather([clip_features, clip_pids, clip_camids, clip_clothes_ids], data_length)

    # Use the averaged feature of all clips split from a video as the representation of this original full-length video
    features = torch.zeros(len(vid2clip_index), clip_features.size(1)).cuda() # torch.Size([316, 8192])
    clip_features = clip_features.cuda() # torch.Size([316, 8192]) torch.Size([25707, 4096])
    pids = torch.zeros(len(vid2clip_index)) # torch.Size([316])
    camids = torch.zeros(len(vid2clip_index)) # torch.Size([316])
    clothes_ids = torch.zeros(len(vid2clip_index)) # torch.Size([316]) 
    for i, idx in enumerate(vid2clip_index):
        features[i] = clip_features[idx[0] : idx[1], :].mean(0)
        features[i] = F.normalize(features[i], p=2, dim=0)
        pids[i] = clip_pids[idx[0]]
        camids[i] = clip_camids[idx[0]]
        clothes_ids[i] = clip_clothes_ids[idx[0]]
    features = features.cpu()

    return features, pids, camids, clothes_ids

@torch.no_grad()
def extract_mevid_vid_feature_with_index(logger, model, dataloader, vid2clip_index, data_length, device, cfg):
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
                write_cache = cfg.MODEL.JEPA_WRITE_IMAGE_CACHE if imgs.dim() == 4 else cfg.MODEL.JEPA_WRITE_TRAIN_CACHE
                jepa_tokens = load_jepa_batch(cfg, jepa_paths, imgs.device, write_cache=write_cache)
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
                feat = model(imgs, clothes_ids, meta)
            
            clip_features.append(feat.cpu())
            clip_pids = torch.cat((clip_pids, pids.cpu()), dim=0)
            clip_camids = torch.cat((clip_camids, camids.cpu()), dim=0)
            clip_clothes_ids = torch.cat((clip_clothes_ids, clothes_ids.cpu()), dim=0)
            clip_index = torch.cat((clip_index, index.cpu()), dim=0)

    clip_features = torch.cat(clip_features, 0)

    # Gather samples from different GPUs
    clip_features, clip_pids, clip_camids, clip_clothes_ids, clip_index = \
        concat_all_gather([clip_features, clip_pids, clip_camids, clip_clothes_ids, clip_index], data_length)

    clip_features = F.normalize(clip_features, p=2, dim=-1)
    clip_features = clip_features.cpu()
    clip_clothes_ids = clip_clothes_ids.cpu()
    return clip_features, clip_pids, clip_camids, clip_clothes_ids, clip_index



def test_mevid(config, model, queryloader, galleryloader, dataset, device, dump_w_index=None, dump=None):
    logger = logging.getLogger('EVA-attribure')
    since = time.time()
    model.eval()
    local_rank = dist.get_rank()
    
    torch.cuda.synchronize()

    
    if dump_w_index:
        qf, q_pids, q_camids, q_clothes_ids, q_index = extract_mevid_vid_feature_with_index(logger, model, queryloader, 
                                dataset.query_vid2clip_index, len(dataset.recombined_query), device, config)
        gf, g_pids, g_camids, g_clothes_ids, g_index = extract_mevid_vid_feature_with_index(logger, model, galleryloader, 
                                dataset.gallery_vid2clip_index, len(dataset.recombined_gallery), device, config)
    else:
        qf, q_pids, q_camids, q_clothes_ids = extract_mevid_vid_feature(logger, model, queryloader, 
                                dataset.query_vid2clip_index, len(dataset.recombined_query), device, config)
        gf, g_pids, g_camids, g_clothes_ids = extract_mevid_vid_feature(logger, model, galleryloader, 
                                dataset.gallery_vid2clip_index, len(dataset.recombined_gallery), device, config)

    # dist.barrier()
    # if local_rank == 0 :
    #     print(f"2 {local_rank} .... ", qf.shape, gf.shape)  
    #     print(f"3 {local_rank} ... ", rearrange(qf, "(B F) (C X)-> B F C X", B=int(qf.shape[0] // 79), F=79, C=128, X=8 ).mean(1).mean(1))
    #     print(f"4 {local_rank} ... ", rearrange(gf, "(B F) (C X)-> B F C X", B=int(gf.shape[0] // 719), F=719, C=128, X=8 ).mean(1).mean(1))
        
    
    # Gather samples from different GPUs
    torch.cuda.empty_cache()
    if dump_w_index:
        qf, q_pids, q_camids, q_clothes_ids, q_index = concat_all_gather([qf, q_pids, q_camids, q_clothes_ids, q_index], len(dataset.query))
        gf, g_pids, g_camids, g_clothes_ids, g_index = concat_all_gather([gf, g_pids, g_camids, g_clothes_ids, g_index], len(dataset.gallery))
    else:
        qf, q_pids, q_camids, q_clothes_ids = concat_all_gather([qf, q_pids, q_camids, q_clothes_ids], len(dataset.query))
        gf, g_pids, g_camids, g_clothes_ids = concat_all_gather([gf, g_pids, g_camids, g_clothes_ids], len(dataset.gallery))
    time_elapsed = time.time() - since
    
    if local_rank == 0:
        logger.info("Extracted features for query set (with different clothes), obtained {} matrix".format(qf.shape))
        logger.info("Extracted features for gallery set, obtained {} matrix".format(gf.shape))
        logger.info('Extracting features complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    # torch.manual_seed(dist.get_rank())
    # qf = torch.rand(316, 1024)
    # gf = torch.rand(1438, 1024)

    # Compute distance matrix between query and gallery
    m, n = qf.size(0), gf.size(0)
    distmat = torch.zeros((m,n)).cuda()
    qf, gf = qf.cuda(), gf.cuda()
    # Cosine similarity
    for i in range(m):
        distmat[i] = (- torch.mm(qf[i:i+1], gf.t()))
    
    # print(dist.get_rank(), distmat.mean(-1)[:10])
    # dist.barrier()
    # dist.broadcast(distmat, src=0)
    # print(dist.get_rank(), distmat.mean(-1)[:10])
    
    distmat = distmat.cpu()
    # np.savez('eval.npz', distmat=distmat, q_pids=q_pids, g_pids=g_pids, q_camids=q_camids, g_camids=g_camids, q_oids=q_clothes_ids, g_oids=g_clothes_ids)
    distmat = distmat.numpy()
    q_pids, q_camids, q_clothes_ids = q_pids.numpy(), q_camids.numpy(), q_clothes_ids.numpy()
    g_pids, g_camids, g_clothes_ids = g_pids.numpy(), g_camids.numpy(), g_clothes_ids.numpy()
    time_elapsed = time.time() - since
    logger.info('Distance computing in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    
    since = time.time()    
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)
    cmc_overall, mAP_overall = cmc, mAP
    if local_rank == 0:    
        logger.info("Computing CMC and mAP")
        logger.info("Overall Results ---------------------------------------------------")
        logger.info('top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
        logger.info("-----------------------------------------------------------")
        time_elapsed = time.time() - since
        logger.info('Using {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    cmc, mAP = evaluate_with_clothes(distmat, q_pids, g_pids, q_camids, g_camids, q_clothes_ids, g_clothes_ids, mode='SC', output=logger.info)
    if local_rank == 0:    
        logger.info("Computing CMC and mAP only for the same clothes setting")
        logger.info("Results ---------------------------------------------------")
        logger.info('top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
        logger.info("-----------------------------------------------------------")

    cmc, mAP = evaluate_with_clothes(distmat, q_pids, g_pids, q_camids, g_camids, q_clothes_ids, g_clothes_ids, mode='CC', output=logger.info)
    cmc_cc, mAP_cc = cmc, mAP
    if local_rank == 0:    
        logger.info("Computing CMC and mAP only for clothes-changing")
        logger.info("Results ---------------------------------------------------")
        logger.info('top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
        logger.info("-----------------------------------------------------------")

    track_scale_file = osp.join(config.DATA.ROOT, 'test_track_scale.txt')
    if osp.exists(track_scale_file):
        cmc, mAP = evaluate_with_locations(distmat, q_pids, g_pids, q_camids, g_camids, mode='SL', config=config, output=logger.info)
        if local_rank == 0:
            logger.info("Computing CMC and mAP only for the same locations setting")
            logger.info("Results ---------------------------------------------------")
            logger.info('top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
            logger.info("-----------------------------------------------------------")

        cmc, mAP = evaluate_with_locations(distmat, q_pids, g_pids, q_camids, g_camids, mode='DL', config=config, output=logger.info)
        cmc_diff_loc, mAP_diff_loc = cmc, mAP
        if local_rank == 0:
            logger.info("Computing CMC and mAP only for locations-changing")
            logger.info("Results ---------------------------------------------------")
            logger.info('top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
            logger.info("-----------------------------------------------------------")

        cmc, mAP = evaluate_with_scales(distmat, q_pids, g_pids, q_camids, g_camids, mode='SS', config=config, output=logger.info)
        if local_rank == 0:
            logger.info("Computing CMC and mAP only for the same scales setting")
            logger.info("Results ---------------------------------------------------")
            logger.info('top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
            logger.info("-----------------------------------------------------------")

        cmc, mAP = evaluate_with_scales(distmat, q_pids, g_pids, q_camids, g_camids, mode='DS', config=config, output=logger.info)
        cmc_diff_scale, mAP_diff_scale = cmc, mAP
        if local_rank == 0:
            logger.info("Computing CMC and mAP only for scales-changing")
            logger.info("Results ---------------------------------------------------")
            logger.info('top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
            logger.info("-----------------------------------------------------------")
    else:
        cmc_diff_loc, mAP_diff_loc = cmc_overall, mAP_overall
        cmc_diff_scale, mAP_diff_scale = cmc_overall, mAP_overall
        if local_rank == 0:
            logger.info("Skipping location/scale MEVID subsets because test_track_scale.txt is missing")

    cmc_acc = dict(cmc_diff_scale=cmc_diff_scale[0], cmc_diff_loc=cmc_diff_loc[0], cmc_cc=cmc_cc[0], cmc_overall=cmc_overall[0], cmc=cmc_overall[0])
    map_acc = dict(mAP_diff_scale=mAP_diff_scale, mAP_diff_loc=mAP_diff_loc, mAP_cc=mAP_cc, mAP_overall=mAP_overall, map=mAP_overall)

    if local_rank == 0:  
        # cmc_diff_scale, mAP_diff_scale
        # cmc_diff_loc, mAP_diff_loc 
        # cmc_cc, mAP_cc 
        # cmc_overall, mAP_overall = cmc, mAP
        if dump_w_index:
            dump_pickle = dict(qf=qf, q_pids=q_pids, q_camids=q_camids, q_clothes_ids=q_clothes_ids, q_index=q_index,
            gf =gf, g_pids=g_pids, g_camids=g_camids, g_clothes_ids=g_clothes_ids, g_index=g_index,
            )
            save_pickle(dump_pickle, config.TAG)
        elif dump:
            dump_pickle = dict(qf=qf, q_pids=q_pids, q_camids=q_camids, q_clothes_ids=q_clothes_ids, 
                gf =gf, g_pids=g_pids, g_camids=g_camids, g_clothes_ids=g_clothes_ids, 
                )
            save_pickle(dump_pickle, config.TAG)

         
    dist.barrier()
    return cmc_overall[0], mAP_overall, cmc_acc, map_acc
    
