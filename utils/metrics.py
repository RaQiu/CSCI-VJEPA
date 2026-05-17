import logging

import torch
import numpy as np
import os

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import norm, multivariate_normal
import torch.distributed as dist
from utils.reranking import re_ranking

from tools.utils import save_pickle

def evaluate_original_cal(distmat, q_pids, g_pids, q_camids, g_camids):
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
    rows_good = np.argwhere(mask == True)
    rows_good = rows_good.flatten()

    cmc[rows_good[0]:] = 1.0
    for i in range(ngood):
        d_recall = 1.0 / ngood
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        ap = ap + d_recall * precision

    return ap, cmc

def euclidean_distance(qf, gf):
    m = qf.shape[0]
    n = gf.shape[0]
    dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mat.addmm_(1, -2, qf, gf.t())
    return dist_mat.cpu().numpy()

def cosine_similarity(qf, gf):
    epsilon = 0.00001
    dist_mat = qf.mm(gf.t())
    qf_norm = torch.norm(qf, p=2, dim=1, keepdim=True)  # mx1
    gf_norm = torch.norm(gf, p=2, dim=1, keepdim=True)  # nx1
    qg_normdot = qf_norm.mm(gf_norm.t())

    dist_mat = dist_mat.mul(1 / qg_normdot).cpu().numpy()
    dist_mat = np.clip(dist_mat, -1 + epsilon, 1 - epsilon)
    dist_mat = np.arccos(dist_mat)
    return dist_mat


def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    num_q, num_g = distmat.shape
    # distmat g
    #    q    1 3 2 4
    #         4 1 2 3
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    #  0 2 1 3
    #  1 2 3 0
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]  # select one row
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        #tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        y = np.arange(1, tmp_cmc.shape[0] + 1) * 1.0
        tmp_cmc = tmp_cmc / y
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP


def eval_func_LTCC(distmat, q_pids, g_pids, q_camids, g_camids, q_clothes_ids, g_clothes_ids, max_rank=50):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    num_q, num_g = distmat.shape

    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))

    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    all_cmc = []
    all_AP = []
    num_valid_q = 0.0  # number of valid query

    for q_idx in range(num_q):
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]
        q_clothid = q_clothes_ids[q_idx]

        order = indices[q_idx]
        # CC
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        remove = remove | ((g_pids[order] == q_pid) & (
                    g_clothes_ids[order] == q_clothid))
        # SC
        # remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        # remove = remove | ((g_pids[order] == q_pid) & ~(g_clothes_ids[order] == q_clothid))

        keep = np.invert(remove)

        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.0

        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        y = np.arange(1, tmp_cmc.shape[0] + 1) * 1.0
        tmp_cmc = tmp_cmc / y
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP


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


class R1_mAP_eval():
    def __init__(self, num_query, max_rank=50, feat_norm=True, reranking=False, length=None):
        super(R1_mAP_eval, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.reranking = reranking
        self.length = length

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []

    def update(self, output):  # called once for each batch
        feat, pid, camid = output
        self.feats.append(feat.cpu())
        self.pids.extend(pid.cpu())
        self.camids.extend(camid.cpu())

    def compute(self):  # called after each epoch
        feats = torch.cat(self.feats, dim=0)
        pids = torch.stack(self.pids, dim=0)
        camids = torch.stack(self.camids, dim=0)

        # print(dist.get_rank(), self.num_query, feats.shape, pids.shape, camids.shape)
        # 0 3543 torch.Size([6927, 1024]) torch.Size([6927]) torch.Size([6927])
        
        # 1 3543 torch.Size([6927, 1024]) torch.Size([6927]) torch.Size([6927])
        # 0 3543 torch.Size([6927, 1024]) torch.Size([6927]) torch.Size([6927])

        feats, pids, camids = concat_all_gather([feats, pids, camids], self.length)
        
        if self.feat_norm:
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)  # along channel
        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(pids[:self.num_query])
        q_camids = np.asarray(camids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(pids[self.num_query:])

        g_camids = np.asarray(camids[self.num_query:])
        if self.reranking:
            print('=> Enter reranking')
            # distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
            distmat = re_ranking(qf, gf, k1=50, k2=15, lambda_value=0.3)

        else:
            print('=> Computing DistMat with euclidean_distance')
            distmat = euclidean_distance(qf, gf)
        # import pdb; pdb.set_trace()
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)
        # cmc, mAP = evaluate_original_cal(distmat, q_pids, g_pids, q_camids, g_camids)
        
        return cmc, mAP, distmat, pids, camids, qf, gf

    def dump_vals(self, indices, name, aux_dump=None):
        feats = torch.cat(self.feats, dim=0)
        pids = torch.stack(self.pids, dim=0)
        camids = torch.stack(self.camids, dim=0)

        print(len(indices), len(feats), len(pids), len(camids), self.length)
        # print(indices)
        feats, pids, camids, indices = concat_all_gather([feats, pids, camids, indices], self.length)
        if self.feat_norm:
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)  # along channel
        print(len(indices), len(feats), len(pids), len(camids), self.length)

        qf = feats[:self.num_query]
        q_pids = np.asarray(pids[:self.num_query])
        q_camids = np.asarray(camids[:self.num_query])
        q_indices = np.asarray(indices[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(pids[self.num_query:])
        g_camids = np.asarray(camids[self.num_query:])
        g_indices = np.asarray(indices[self.num_query:])

        distmat = euclidean_distance(qf, gf)
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)
        print(cmc[0], mAP)

        feature_dump = dict(
            qf=qf.cpu(), q_pids=q_pids, q_camids=q_camids, q_image_paths=q_indices,
            gf=gf.cpu(), g_pids=g_pids, g_camids=g_camids, g_image_paths=g_indices,
        )
        if aux_dump:
            feature_dump.update(aux_dump)
        
        save_pickle(feature_dump, name)
        print(f".... Dumping is complete @ {name}")

class R1_mAP_eval_LTCC():
    def __init__(self, num_query, max_rank=50, feat_norm=True, reranking=False, length=None):
        super(R1_mAP_eval_LTCC, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.reranking = reranking
        self.length = length


    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []
        self.cloth_ids = []

    def update(self, output):  # called once for each batch
        feat, pid, camid, cloth_id = output
        self.feats.append(feat.cpu())
        self.pids.extend(pid.cpu())
        self.camids.extend(camid.cpu())
        self.cloth_ids.extend(cloth_id.cpu())

    def compute(self):  # called after each epoch
        feats = torch.cat(self.feats, dim=0)
        pids = torch.stack(self.pids, dim=0)
        camids = torch.stack(self.camids, dim=0)
        clothids = torch.stack(self.cloth_ids, dim=0)

        feats, pids, camids, cloth_ids= concat_all_gather([feats, pids, camids, clothids], self.length)

        if self.feat_norm:
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)  # along channel
        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(pids[:self.num_query])
        q_camids = np.asarray(camids[:self.num_query])
        q_clothes_ids = np.asarray(cloth_ids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])

        g_camids = np.asarray(camids[self.num_query:])
        g_clothes_ids = np.asarray(cloth_ids[self.num_query:])

        # print(qf.mean(-1), print(gf.mean(-1)))
        if self.reranking:
            print('=> Enter reranking')
            # distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
            distmat = re_ranking(qf, gf, k1=50, k2=15, lambda_value=0.3)

        else:
            print('=> Computing DistMat with euclidean_distance')
            distmat = euclidean_distance(qf, gf)
        # print(distmat.shape, distmat.mean(-1)[:10], q_pids[:30], g_pids[:30], q_camids[:30], g_camids[:30], q_clothes_ids[:30], g_clothes_ids[:30])

        cmc, mAP = eval_func_LTCC(distmat, q_pids, g_pids, q_camids, g_camids,q_clothes_ids,g_clothes_ids)

        return cmc, mAP, distmat, pids, camids, qf, gf

    def dump_vals(self, indices, name, aux_dump=None):
        feats = torch.cat(self.feats, dim=0)
        pids = torch.stack(self.pids, dim=0)
        camids = torch.stack(self.camids, dim=0)
        clothids = torch.stack(self.cloth_ids, dim=0)

        print(len(indices), len(feats), len(pids), len(camids), self.length)
        # print(indices)
        feats, pids, camids, indices, cloth_ids = concat_all_gather([feats, pids, camids, indices, clothids], self.length)
        if self.feat_norm:
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)  # along channel
        print(len(indices), len(feats), len(pids), len(camids), self.length)

        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(pids[:self.num_query])
        q_camids = np.asarray(camids[:self.num_query])
        q_clothes_ids = np.asarray(cloth_ids[:self.num_query])
        q_indices = np.asarray(indices[:self.num_query])


        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(camids[self.num_query:])
        g_clothes_ids = np.asarray(cloth_ids[self.num_query:])
        g_indices = np.asarray(indices[self.num_query:])

        distmat = euclidean_distance(qf, gf)
        print(distmat.shape, distmat.mean(-1)[:10], q_pids[:30], g_pids[:30], q_camids[:30], g_camids[:30], q_clothes_ids[:30], g_clothes_ids[:30])
        cmc, mAP = eval_func_LTCC(distmat, q_pids, g_pids, q_camids, g_camids,q_clothes_ids,g_clothes_ids)
        print(cmc[0], mAP)

        feature_dump = dict(
            qf=qf.cpu(), q_pids=q_pids, q_camids=q_camids, q_image_paths=q_indices, q_clothes_ids=q_clothes_ids,
            gf=gf.cpu(), g_pids=g_pids, g_camids=g_camids, g_image_paths=g_indices, g_clothes_ids=g_clothes_ids, 
        )
        if aux_dump:
            feature_dump.update(aux_dump)
        
        
        save_pickle(feature_dump, name)
        print(f".... Dumping is complete @ {name}")


class R1_mAP_eval_LaST(R1_mAP_eval):
    def __init__(self, q_length=None, g_length=None, **args):
        super(R1_mAP_eval_LaST, self).__init__(**args)
        
        self.q_length = q_length
        self.g_length = g_length

    def reset(self):
        self.q_feats = []
        self.q_pids = []
        self.q_camids = [] 

        self.g_feats = []
        self.g_pids = []
        self.g_camids = [] 

    def update(self, output, query=True):  # called once for each batch
        feat, pid, camid = output
        if query:
            self.q_feats.append(feat.cpu())
            self.q_pids.extend(pid.cpu())
            self.q_camids.extend(camid.cpu())
        else:
            self.g_feats.append(feat.cpu())
            self.g_pids.extend(pid.cpu())
            self.g_camids.extend(camid.cpu())

    def compute(self):  # called after each epoch
        
        qf = torch.cat(self.q_feats, dim=0)
        q_pids = torch.stack(self.q_pids, dim=0)
        q_camids = torch.stack(self.q_camids, dim=0)
    
        gf = torch.cat(self.g_feats, dim=0)
        g_pids = torch.stack(self.g_pids, dim=0)
        g_camids = torch.stack(self.g_camids, dim=0)
    
        # print(dist.get_rank(), len(qf), len(q_pids), len(q_camids))
        # 0 5088 5088 5088 # 1 5088 5088 5088
        # print(dist.get_rank(), len(gf), len(g_pids), len(g_camids))
        # 0 62677 62677 62677 # 1 62677 62677 62677

        # print(dist.get_rank(), len(dataset.gallery))
        # 0 125353 1254

        qf, q_pids, q_camids = concat_all_gather([qf, q_pids, q_camids], self.q_length)
        gf, g_pids, g_camids = concat_all_gather([gf, g_pids, g_camids], self.g_length)

        # print(dist.get_rank(), qf.shape, q_pids.shape, q_camids.shape, self.q_length)
        # 0 torch.Size([10176, 1024]) torch.Size([10176]) torch.Size([10176]) 10176  # 1 torch.Size([10176, 1024]) torch.Size([10176]) torch.Size([10176]) 10176
        # print(dist.get_rank(), gf.shape, g_pids.shape, g_camids.shape, self.g_length)
        # 0 torch.Size([125353, 1024]) torch.Size([125353]) torch.Size([125353]) 125353 # 1 torch.Size([125353, 1024]) torch.Size([125353]) torch.Size([125353]) 125353

        if self.feat_norm:
            print("The test feature is normalized")
            qf = torch.nn.functional.normalize(qf, dim=1, p=2)  # along channel
            gf = torch.nn.functional.normalize(gf, dim=1, p=2)  # along channel

        # query
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)
        
        # gallery
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)
        
        if self.reranking:
            print('=> Enter reranking')
            # distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
            distmat = re_ranking(qf, gf, k1=50, k2=15, lambda_value=0.3)

        else:
            print('=> Computing DistMat with euclidean_distance')
            distmat = euclidean_distance(qf, gf)
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)

        return cmc, mAP, distmat, None, None, qf, gf

    def dump_vals(self, indices, name):
         
        qf = torch.cat(self.q_feats, dim=0)
        q_pids = torch.stack(self.q_pids, dim=0)
        q_camids = torch.stack(self.q_camids, dim=0)
    
        gf = torch.cat(self.g_feats, dim=0)
        g_pids = torch.stack(self.g_pids, dim=0)
        g_camids = torch.stack(self.g_camids, dim=0)
    
        qf, q_pids, q_camids = concat_all_gather([qf, q_pids, q_camids], self.q_length)
        gf, g_pids, g_camids = concat_all_gather([gf, g_pids, g_camids], self.g_length)

        assert False, "not yet verified  with indicies yet and dumping"


        print(len(indices), len(feats), len(pids), len(camids), self.length)
        # print(indices)
        feats, pids, camids = concat_all_gather([feats, pids, camids], self.length)
        print(len(indices), len(feats), len(pids), len(camids), self.length)

        qf = feats[:self.num_query]
        q_pids = np.asarray(pids[:self.num_query])
        q_camids = np.asarray(camids[:self.num_query])
        q_indices = np.asarray(indices[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(pids[self.num_query:])
        g_camids = np.asarray(camids[self.num_query:])
        g_indices = np.asarray(indices[self.num_query:])

        feature_dump = dict(
            qf=qf.cpu(), q_pids=q_pids, q_camids=q_camids, q_image_paths=q_indices,
            gf=gf.cpu(), g_pids=g_pids, g_camids=g_camids, g_image_paths=g_indices,
        )
        save_pickle(feature_dump, name)
        






def polychor(x, y=None, ML=False, std_err=False, maxcor=0.9999, start=None, thresholds=False):
    def f(pars):
        pars = np.atleast_1d(pars)
        rho = pars[0]
        rho = np.clip(rho, -maxcor, maxcor)
        if len(pars) == 1:
            row_cuts = rc
            col_cuts = cc
        else:
            row_cuts = pars[1:r]
            col_cuts = pars[r:r+c-1]
            if any(np.diff(row_cuts) < 0) or any(np.diff(col_cuts) < 0):
                return np.inf
        P = binBvn(rho, row_cuts, col_cuts)
        return -np.sum(tab * np.log(P + 1e-6))

    if y is None:
        tab = x
    else:
        tab = np.histogram2d(x, y, bins=[4, 4])[0] 
        
    zerorows = np.all(tab == 0, axis=1)
    zerocols = np.all(tab == 0, axis=0)
    zr = np.sum(zerorows)
    zc = np.sum(zerocols)
    
    if zr > 0:
        print(f"{zr} rows with zero marginal removed")
    if zc > 0:
        print(f"{zc} columns with zero marginal removed")
        
    tab = tab[~zerorows, :]
    tab = tab[:, ~zerocols]
    r, c = tab.shape
    
    if r < 2:
        print("The table has fewer than 2 rows")
        return None
    if c < 2:
        print("The table has fewer than 2 columns")
        return None
    
    n = np.sum(tab)
    rc = norm.ppf(np.cumsum(np.sum(tab, axis=1)) / n)[:-1]
    cc = norm.ppf(np.cumsum(np.sum(tab, axis=0)) / n)[:-1]
    
    if start is not None and (ML or std_err):
        if isinstance(start, dict):
            rho = start['rho']
            rc = start['row_thresholds']
            cc = start['col_thresholds']
        else:
            rho = start
        if not isinstance(rho, (int, float)) or len(np.atleast_1d(rho)) != 1:
            raise ValueError("Start value for rho must be a number")
        if not isinstance(rc, np.ndarray) or len(rc) != r - 1:
            raise ValueError("Start values for row thresholds must be r - 1 numbers")
        if not isinstance(cc, np.ndarray) or len(cc) != c - 1:
            raise ValueError("Start values for column thresholds must be c - 1 numbers")
    
    if ML:
        if start is None:
            rho = minimize_scalar(f).x
            initial_guess = np.concatenate(([rho], rc, cc))
        else:
            initial_guess = np.concatenate(([rho], rc, cc))
        result = minimize(f, initial_guess, method='L-BFGS-B')
        rho = result.x[0]
        rho = np.clip(rho, -maxcor, maxcor)
        
        if std_err:
            chisq = 2 * (result.fun + np.sum(tab * np.log((tab + 1e-6) / n)))
            df = len(tab) - r - c
            return {
                'type': 'polychoric',
                'rho': rho,
                'row_cuts': result.x[1:r],
                'col_cuts': result.x[r:r+c-1],
                'var': np.linalg.inv(result.hess_inv.todense()),
                'n': n,
                'chisq': chisq,
                'df': df,
                'ML': True
            }
        elif thresholds:
            return {
                'type': 'polychoric',
                'rho': rho,
                'row_cuts': result.x[1:r],
                'col_cuts': result.x[r:r+c-1],
                'var': None,
                'n': n,
                'chisq': None,
                'df': None,
                'ML': True
            }
        else:
            return rho
        
    elif std_err:
        result = minimize(f, [0], method='BFGS')
        rho = result.x[0]
        rho = np.clip(rho, -maxcor, maxcor)
        chisq = 2 * (result.fun + np.sum(tab * np.log((tab + 1e-6) / n)))
        df = len(tab) - r - c
        return {
            'type': 'polychoric',
            'rho': rho,
            'row_cuts': rc,
            'col_cuts': cc,
            'var': 1 / result.hess_inv[0, 0],
            'n': n,
            'chisq': chisq,
            'df': df,
            'ML': False
        }
    else:
        rho = minimize_scalar(f).x
        if thresholds:
            return {
                'type': 'polychoric',
                'rho': rho,
                'row_cuts': rc,
                'col_cuts': cc,
                'var': None,
                'n': n,
                'chisq': None,
                'df': None,
                'ML': False
            }
        else:
            return rho


def binBvn(rho, row_cuts, col_cuts, bins=4):
    row_cuts = np.concatenate(([-np.inf], row_cuts, [np.inf])) if row_cuts is not None else np.concatenate(([-np.inf], np.linspace(0, 1, bins)[1:], [np.inf]))
    col_cuts = np.concatenate(([-np.inf], col_cuts, [np.inf])) if col_cuts is not None else np.concatenate(([-np.inf], np.linspace(0, 1, bins)[1:], [np.inf]))
    r = len(row_cuts) - 1
    c = len(col_cuts) - 1
    P = np.zeros((r, c))

    for i in range(r):
        for j in range(c):
            lower = np.array([row_cuts[i], col_cuts[j]])
            upper = np.array([row_cuts[i+1], col_cuts[j+1]])

            # Check for invalid input values
            if np.any(~np.isfinite(lower)) or np.any(~np.isfinite(upper)):
                continue

            # Calculate the multivariate normal CDF
            P[i, j] = multivariate_normal.cdf(upper, mean=[0, 0], cov=[[1, rho], [rho, 1]]) - \
                      multivariate_normal.cdf([lower[0], upper[1]], mean=[0, 0], cov=[[1, rho], [rho, 1]]) - \
                      multivariate_normal.cdf([upper[0], lower[1]], mean=[0, 0], cov=[[1, rho], [rho, 1]]) + \
                      multivariate_normal.cdf(lower, mean=[0, 0], cov=[[1, rho], [rho, 1]])

            # Check for invalid output values
            if not np.isfinite(P[i, j]):
                P[i, j] = 0

    return P
