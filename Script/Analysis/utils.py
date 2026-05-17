import os
import pickle
import time
import torch 
import numpy as np 
from PIL import Image
from einops import rearrange, repeat

def load_pickle(name):
    # Load data (deserialize)
    with open(f'{name}.pkl', 'rb') as handle:
        data = pickle.load(handle)
    return data


def make_folder(name):
    try: 
        os.mkdir(name) 
    except OSError as error: 
        _ = 0 

def euclidean_distance(qf, gf):
    m = qf.shape[0]
    n = gf.shape[0]
    dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mat.addmm_(1, -2, qf, gf.t())
    return dist_mat.cpu().numpy()


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
    
    return distmat, time_elapsed

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



def compute_acc(index, good_index, junk_index):
    cmc = np.zeros(len(index)) 
    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]
    # find good_index index
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask==True)
    rows_good = rows_good.flatten()
    if rows_good[0] !=  0:
        return False, index[0]
    return True, index[0]  
    

def misfit_ltcc(dismat, g_pids, q_pids, g_camids, q_camids, g_clothes_ids, q_clothes_ids):
    num_q, num_g = dismat.shape
    index = np.argsort(dismat, axis=1) # from small to large
    CMC = np.zeros(len(g_pids))
    mode = "CC"
    count = 0 
    r1 = 0
    misfits = []
    corrects = []
    for i in range(num_q):
        # groundtruth index
        query_index = np.argwhere(g_pids==q_pids[i])
        camera_index = np.argwhere(g_camids==q_camids[i])
        cloth_index = np.argwhere(g_clothes_ids==q_clothes_ids[i])
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
            continue
        count += 1
        correct , mis_fit = compute_acc(index[i], good_index, junk_index)
        if correct:
            r1 += 1
            corrects.append([i, mis_fit])
        else:
            misfits.append([i, mis_fit])
    return r1, misfits, corrects, count

# ltcc, celeb. deepchange
def simple_identifier(img_path):
    indentifier = img_path.split("/")[-1][:-4]
    return indentifier

def mevid_indentifier(path):
    return path.split("/")[-1][:-4]

def ccvid_indentifier(path):
    return "_".join(path.split("/")[-3:])[:-4]    

class HISTOGRAM_CLASS():
    def __init__(self, color_profile):
        from data.rgbuc import RGBuvHistBlock, RGBuvHistBlock_Original
        import torchvision.transforms as transforms

        self.color_profile = color_profile
        if color_profile in [3, 42, 21, 16, 13, 9, 12, 35, 47, 5, 14, 44, 43, 39, 38, 34, 18, 17, 2, 18]:
            self.fix_loader, fn_str = self.w_color, "w_color"
        elif color_profile in [24, 25, 40, 41, 49, 23, 32, 28, 48, 29, 27, 26]:
            self.fix_loader, fn_str = self.w_color3, "w_color3"
        elif color_profile in [50, 51, 52, 53, 54, 55, 56, 57]:
            self.fix_loader, fn_str = self.w_color2, "w_color2"

        if color_profile in [3, 42, 24, 25, 40, 41, 49, 23, 32, 48, 34, 18, 17, 2]:
            self.mode, self.p = "norm", -1
        elif color_profile in [21, 5, 38, 27, 26, 50, 51, 52, 53]:
            self.mode, self.p = "l2", 2
        elif color_profile in [16, 13, 9, 12, 35, 47, 28, 14, 44, 43, 39, 29, 54, 55, 56, 57]:
            self.mode, self.p = "l1", 1
        
        if color_profile in [3, 42, 25, 21, 9, 35, 28, 43, 38, 34, 18, 54, 50, ]:
            self.wt = 10
        elif color_profile in [24, 40, 41, 49, 23, 16, 12, 47, 48, 39, 29, 27, 2, 55, 51]:
            self.wt = 1000
        elif color_profile in [32, 13, 44, 17, 56, 52]:
            self.wt = 1
        elif color_profile in [5, 14, 26, 57, 53]:
            self.wt = 100
        
        if color_profile in [3, 25, 24, 23, 32, 21, 16, 13, 9, 12, 28, 5, 14, 29, 27, 26, 18, 17, 2, 50, 51, 52, 53, 54, 55, 56, 57]:
            dim = 32
        elif color_profile in [42, 49, 47, 48, 44, 43]:
            dim = 64
        elif color_profile in [40, 41, 35, 39, 38, 34]:
            dim = 16

        if color_profile in [3, 42, 24, 25, 40, 9, 12, 35, 47, 28, 5, 48, 43, 39, 34, 29, 27, 26, 2]:
            sigma, sigma_fn = True , "0.001"
        elif color_profile in [41, 49, 23, 32, 21, 16, 13, 14, 44, 38, 18, 17, 50, 51, 52, 53, 54, 55, 56, 57]:
            sigma, sigma_fn = False , "N/A"

        
        if sigma == True:
            self.histblock = RGBuvHistBlock(insz=224, h=dim,  intensity_scale=False,  method='inverse-quadratic', device='cpu', sigma=0.001)
            print(f"RGBuvHistBlock(insz=224, h={dim},  intensity_scale=False,  method='inverse-quadratic', device='cpu', sigma=0.001)")
        else:
            self.histblock = RGBuvHistBlock(insz=224, h=dim,  intensity_scale=False,  method='inverse-quadratic', device='cpu')
            print(f"RGBuvHistBlock(insz=224, h={dim},  intensity_scale=False,  method='inverse-quadratic', device='cpu')")
        print(f"{self.mode}:{self.p}, {self.wt}, {fn_str}, {sigma_fn}")
        
        self.loader = transforms.Compose([transforms.ToTensor()]) 
    def w_color(self, img):
        hist_image = self.histblock(img.unsqueeze(0))
        hist_image = hist_image.mean(1) 
        hist_image = rearrange(hist_image, "B H W -> B (H W)") 
        if self.mode == "l2" or self.mode == "l1": 
            hist_image = torch.nn.functional.normalize(hist_image.float() , p=self.p, dim=-1)
        else:
            hist_image = normalize(hist_image)
        hist_image = hist_image.float() * self.wt
        return hist_image

    def w_color3(self, img):
        hist_image = self.histblock(img.unsqueeze(0))
        hist_image = rearrange(hist_image, "B C H W -> B (C H W)")         
        if self.mode == "l2" or self.mode == "l1": 
            hist_image = torch.nn.functional.normalize(hist_image.float() , p=self.p, dim=-1)
        else:
            hist_image = normalize(hist_image)
        hist_image = hist_image.float() * self.wt
        return hist_image

    def w_color2(self, img):
        hist_image = cv2.calcHist([ denormalize.permute(1,2,0).numpy() ],[0,1,2],None,[20,20,20],[0,1,0,1,0,1])
        hist_image = torch.tensor(hist_image)
        hist_image = rearrange(hist_image, "C H W -> (C H W)") 
        if self.mode == "l2" or self.mode == "l1": 
            hist_image = torch.nn.functional.normalize(hist_image.float() , p=self.p, dim=-1)
        else:
            hist_image = normalize(hist_image)
        hist_image = hist_image.float() * self.wt
        return hist_image

    def __call__(self, img):
        image = self.loader(Image.open( img ).resize( (224,224)) ) 
        histogram = self.fix_loader(image)
        return histogram
        