import torch
import functools
import os.path as osp
import os 
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import random 
from tools.utils import load_pickle
from torchvision.utils import save_image 
import pandas as pd 
from einops import rearrange
from functools import partial 
from data.rgbuc import RGBuvHistBlock, RGBuvHistBlock_Original
from data.img_transforms import UnNormalize
from tools.utils import normalize, colorname_to_rgb, create_low_res, create_blur_motion, create_g_blur

try:
    from mmcv.fileio import FileClient
except:
    from mmengine.fileio import FileClient
try:
    from mmcv.runner.utils import set_random_seed
except:
    from mmengine.runner import set_random_seed

io_backend='disk'
file_client=None
try:
    import decord
    import io 
    file_client = FileClient(io_backend)
except:
    print("Decod / file_client  failed, likely multi node code")

import cv2


def video_without_img_paths(video_path, num_threads=1):
    file_obj = io.BytesIO(file_client.get(video_path))
    container = decord.VideoReader(file_obj, num_threads=num_threads)
    return container

def read_image(img_path, grey_scale=None ):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    if grey_scale:
        img = img.convert('L').convert(mode='RGB')
        # img.save("temp.png")
    return img


class ImageDataset(Dataset):
    """Image Person ReID Dataset"""
    def __init__(self, dataset, aux_info=False, transform=None, train=None, Debug=None, return_index=None, datatset_name=None, grey_scale=None, **kwargs ):
        self.dataset = dataset
        self.transform = transform
        self.aux_info = aux_info
        self.train = train
        self.Debug = Debug
        self.return_index = return_index
        self.datatset_name = datatset_name
        self.grey_scale = grey_scale
        if self.dataset and len(self.dataset):self.__getitem__(0)
        

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, clothes_id,aux_info = self.dataset[index]
        img = read_image(img_path, self.grey_scale)
        cloth_id_batch = torch.tensor(clothes_id, dtype=torch.int64)
        if self.transform is not None:
            img = self.transform(img)
        if type(aux_info) is np.ndarray:
            select_index = np.random.randint(aux_info.shape[0])
            if self.return_index:
                return img, pid,camid, clothes_id,cloth_id_batch, aux_info[select_index,:], index
            return img, pid,camid, clothes_id,cloth_id_batch, aux_info[select_index,:]
        else:
            if self.return_index:
                return img, pid,camid, clothes_id,cloth_id_batch, np.asarray(aux_info).astype(np.float64), index
            return img, pid,camid, clothes_id,cloth_id_batch, np.asarray(aux_info).astype(np.float64)
        
        
class Video_as_Image(ImageDataset):
    def train_loader(self, index):
        img_path, pid, camid, clothes_id,aux_info = self.dataset[index]

        img_path = random.choice(img_path)
        img = read_image(img_path)
        img = self.transform(img)
        cloth_id_batch = torch.tensor(clothes_id, dtype=torch.int64) 
        return img, pid,camid, clothes_id,cloth_id_batch, np.asarray(aux_info).astype(np.float64)

    def test_loader(self, index):
        img_path, pid, camid, clothes_id,aux_info = self.dataset[index]
        img_path = img_path[len(img_path) // 2]
        img = read_image(img_path)
        # img.save("temp.png")
        img = self.transform(img)
        # save_image (img , "temp.png")
        cloth_id_batch = torch.tensor(clothes_id, dtype=torch.int64) 
        if self.return_index:
            return img, pid,camid, clothes_id,cloth_id_batch, np.asarray(aux_info).astype(np.float64), index
        return img, pid,camid, clothes_id,cloth_id_batch, np.asarray(aux_info).astype(np.float64)

    def __getitem__(self, index):
        if self.train:
            return self.train_loader(index)    
        else:    
            return self.test_loader(index)    

class Video_as_Image_fixes(Video_as_Image):
    def __init__(self, color_adv=None, transform=None, train=None, datatset_name=None, need_fixing=None, color_profile=None, **kwargs):
        self.fix_loader = None 
        self.return_colors = None
        self.datatset_name = datatset_name
        self.need_fixing = need_fixing
        
        self.aug_setup(color_adv, train, transform, color_profile=color_profile)

        if "prcc" in datatset_name:
            self.indentifier_fn = self.prcc_indentifier
        else:
            self.indentifier_fn = self.simple_identifier

        super().__init__( transform=transform, train=train, datatset_name=datatset_name, **kwargs)

    def simple_identifier(self, img_path):
        indentifier = img_path.split("/")[-1][:-4]
        return indentifier

    def prcc_indentifier(self, img):
        id, image_name = img.split("/")[-2:]
        identifier = id + "_" + image_name[:-4]
        return identifier 
        # identifier = f"{session}_{folder}_{image_name[:-4]}.png"
            
    def aug_setup(self, color_adv, train, transform, color_profile=None):
        assert color_adv and color_profile is not None 
    
        self.return_colors = True 
        # self.fix_loader = self.w_color2
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
        elif color_profile in [50, 51, 52, 53, 54, 55, 56, 57]:
            self.histblock = None 
            print(f"RGB Histogram h={dim})")
        else:
            self.histblock = RGBuvHistBlock(insz=224, h=dim,  intensity_scale=False,  method='inverse-quadratic', device='cpu')
            print(f"RGBuvHistBlock(insz=224, h={dim},  intensity_scale=False,  method='inverse-quadratic', device='cpu')")

    
        print(f"{self.mode}:{self.p}, {self.wt}, {fn_str}, {sigma_fn}")
        # self.histblock = RGBuvHistBlock_Original(insz=224, h=32,  intensity_scale=False,  method='RBF', device='cpu', sigma=0.001)
        # print("RGBuvHistBlock_Original(insz=224, h=32,  intensity_scale=False,  method='RBF', device='cpu', sigma=0.001)")
        self.need_fixing = True 

        if train:
            normalize = transform.transforms[-2]
            self.de_normalize = UnNormalize(mean=normalize.mean, std=normalize.std)
    
    def w_color(self, img_path):
        img = read_image(img_path)
        # img.save("temp0.png")
        
        img = self.transform(img)
        # save_image(img, "temp.png")
        denormalize = self.de_normalize(img)
        # save_image(denormalize, "temp2.png")

        hist_image = self.histblock(denormalize.unsqueeze(0))
        # save_image(hist_image * 100, "temp3.png")

        hist_image = hist_image.mean(1) 
        # save_image(hist_image.unsqueeze(1) * 100, "temp4.png")

        hist_image = rearrange(hist_image, "B H W -> B (H W)") 
        if self.mode == "l2" or self.mode == "l1": 
            hist_image = torch.nn.functional.normalize(hist_image.float() , p=self.p, dim=-1)
        else:
            hist_image = normalize(hist_image)
        hist_image = hist_image.float() * self.wt
        # print(hist_image.max(), hist_image.min(), hist_image.mean())
        return img , hist_image

    def w_color3(self, img_path):
        img = read_image(img_path)
        # img.save("temp0.png")
        
        img = self.transform(img)
        # save_image(img, "temp.png")
        denormalize = self.de_normalize(img)
        # save_image(denormalize, "temp2.png")

        hist_image = self.histblock(denormalize.unsqueeze(0))
        hist_image = rearrange(hist_image, "B C H W -> B (C H W)") 
        
        if self.mode == "l2" or self.mode == "l1": 
            hist_image = torch.nn.functional.normalize(hist_image.float() , p=self.p, dim=-1)
        else:
            hist_image = normalize(hist_image)
        hist_image = hist_image.float() * self.wt
        # print(hist_image.max(), hist_image.min(), hist_image.mean())
        return img , hist_image

    def w_color2(self, img_path):
        img = read_image(img_path)
        # img.save("temp0.png")
        
        img = self.transform(img)
        # save_image(img, "temp.png")
        denormalize = self.de_normalize(img)
        # save_image(denormalize, "temp2.png")

        hist_image = cv2.calcHist([ denormalize.permute(1,2,0).numpy() ],[0,1,2],None,[20,20,20],[0,1,0,1,0,1])
        hist_image = torch.tensor(hist_image)
        hist_image = rearrange(hist_image, "C H W -> (C H W)") 
        
        if self.mode == "l2" or self.mode == "l1": 
            hist_image = torch.nn.functional.normalize(hist_image.float() , p=self.p, dim=-1)
        else:
            hist_image = normalize(hist_image)
        hist_image = hist_image.float() * self.wt

        
        return img , hist_image

    def fixed_fn (self, img_path):
        if self.return_colors :
            img , color_label = self.fix_loader(img_path)
            return img , color_label
        else:
            img = self.fix_loader(img_path)
            return img, None  

    def train_loader(self, index):
        img_path, pid, camid, clothes_id,aux_info = self.dataset[index]

        img_path = random.choice(img_path)
        img , extra_data = self.fixed_fn (img_path)    

        cloth_id_batch = torch.tensor(clothes_id, dtype=torch.int64)
        if self.return_colors :
            return img, pid,camid, clothes_id,cloth_id_batch, np.asarray(aux_info).astype(np.float64), extra_data    
        return img, pid,camid, clothes_id,cloth_id_batch, np.asarray(aux_info).astype(np.float64)

class ImageDataset_fixes(Video_as_Image_fixes):
    def train_loader(self, index):
        img_path, pid, camid, clothes_id,aux_info = self.dataset[index]
        
        img , extra_data = self.fixed_fn (img_path)    
        
        cloth_id_batch = torch.tensor(clothes_id, dtype=torch.int64)
        if self.return_colors :
            return img, pid,camid, clothes_id,cloth_id_batch, np.asarray(aux_info).astype(np.float64), extra_data    
        return img, pid,camid, clothes_id,cloth_id_batch, np.asarray(aux_info).astype(np.float64)

    def test_loader(self, index):
        img_path, pid, camid, clothes_id,aux_info = self.dataset[index]
        img = read_image(img_path)
        img = self.transform(img)
        cloth_id_batch = torch.tensor(clothes_id, dtype=torch.int64) 
        if self.return_index:
            return img, pid,camid, clothes_id,cloth_id_batch, np.asarray(aux_info).astype(np.float64), index
        return img, pid,camid, clothes_id,cloth_id_batch, np.asarray(aux_info).astype(np.float64)



    

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def accimage_loader(path):
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def get_default_image_loader():
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader

def image_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

def video_loader(img_paths, image_loader):
    video = []
    for image_path in img_paths:
        if osp.exists(image_path):
            video.append(image_loader(image_path))
        else:
            return video

    return video

def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)


class VideoDataset(Dataset):
    """Video Person ReID Dataset.
    Note:
        Batch data has shape N x C x T x H x W
    Args:
        dataset (list): List with items (img_paths, pid, camid)
        temporal_transform (callable, optional): A function/transform that  takes in a list of frame indices
            and returns a transformed version
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an video given its path and frame indices.
    """

    def __init__(self, 
                 dataset, 
                 spatial_transform=None,
                 temporal_transform=None,
                 get_loader=get_default_video_loader, 
                 train=None, 
                 cloth_changing=True, F8= None, Debug=None, return_index=None,
                 return_jepa_paths=None, **args):
        self.dataset = dataset
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.loader = get_loader()
        self.cloth_changing = cloth_changing
        self.train = train
        self.F8 = F8
        self.Debug = Debug
        self.return_index = return_index
        self.return_jepa_paths = return_jepa_paths
        self.__getitem__(0)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_paths, pid, camid, clothes_id, attr = self.dataset[index]
        selected_img_paths = list(img_paths)
        if self.train:
            selected_img_paths = self.temporal_transform(selected_img_paths)
        elif not self.F8:
            selected_img_paths = selected_img_paths[::2]

        clip = self.loader(selected_img_paths)
        
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]

        clip = torch.stack(clip, 0)
        
        # trans T x C x H x W to C x T x H x W
        clip = clip.permute(1, 0, 2, 3)

        
        cloth_id_batch = torch.tensor(clothes_id, dtype=torch.int64) 
        jepa_paths = "\n".join(selected_img_paths)
        if self.return_index:
            if self.return_jepa_paths:
                return clip, pid, camid, clothes_id, cloth_id_batch, np.asarray(attr).astype(np.float64), index, jepa_paths
            return clip, pid, camid, clothes_id, cloth_id_batch, np.asarray(attr).astype(np.float64) , index     
        if self.return_jepa_paths:
            return clip, pid, camid, clothes_id, cloth_id_batch, np.asarray(attr).astype(np.float64), jepa_paths
        return clip, pid, camid, clothes_id, cloth_id_batch, np.asarray(attr).astype(np.float64) 
                
    def set_epoch(self, epoch):
        self.epoch = epoch
    
