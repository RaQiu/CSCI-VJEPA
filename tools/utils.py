import os
import sys
import shutil
import errno
import json
import os.path as osp
import torch
import random
import logging
import numpy as np
import pickle
from einops import rearrange, repeat
from torchvision.utils import save_image 
import matplotlib.colors as mcolors

import torchvision.transforms as transforms
import cv2
from PIL import Image

def set_seed(seed=None):
    if seed is None:
        return
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = ("%s" % seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def read_json(fpath):
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))


class AverageMeter(object):
    """Computes and stores the average and current value.
       
       Code imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, fpath='checkpoint.pth.tar'):
    mkdir_if_missing(osp.dirname(fpath))
    torch.save(state, fpath)
    if is_best:
        shutil.copy(fpath, osp.join(osp.dirname(fpath), 'best_model.pth.tar'))

'''
class Logger(object):
    """
    Write console output to external text file.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()
'''


def get_logger(fpath, local_rank=0, name=''):
    # Creat logger
    logger = logging.getLogger(name)
    level = logging.INFO if local_rank in [-1, 0] else logging.WARN
    logger.setLevel(level=level)

    # Output to console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level=level) 
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console_handler)

    # Output to file
    if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
    file_handler = logging.FileHandler(fpath, mode='w')
    file_handler.setLevel(level=level)
    file_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(file_handler)

    return logger



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

def save_pickle(data, name):
    # Store data (serialize)
    with open(f'{name}.pkl', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)




def normalize(x):
    return (x - x.min()) / (x.max() - x.min())

def rearrange_mlr(x):
    x = rearrange(x, "B N ... -> (B N) ... ")
    return x

def expand_input(x, N):
    x = repeat(x, "B ... -> B N ... ", N= N)
    return rearrange_mlr(x)

def reverse_arrange(x, B, N):
    x = rearrange(x, "(B N) ... -> B N ... ", B=B, N= N)
    return x


def gif_generator(x, name="array.gif"):
    # x = [PIL images ]
    images = [] 
    for img in x:y=img;images.append(y)
    images[0].save(name, save_all=True, append_images=images[1:], duration= len(x) / 30 , loop=0)
    # images[0].save(name, save_all=True, append_images=images[1:], duration=1000/30, loop=0)


def colorname_to_rgb(colorname):
    rgb_float = mcolors.to_rgb(colorname)
    rgb_uint8 = tuple((np.array(rgb_float)*255).astype(np.uint8))
    return rgb_uint8


def apply_motion_blur(image, size, angle):
    k = np.zeros((size, size), dtype=np.float32)
    k[ (size-1)// 2 , :] = np.ones(size, dtype=np.float32)
    k = cv2.warpAffine(k, cv2.getRotationMatrix2D( (size / 2 -0.5 , size / 2 -0.5 ) , angle, 1.0), (size, size) )  
    k = k * ( 1.0 / np.sum(k) )        
    return cv2.filter2D(image, -1, k) 


def create_low_res(img_hr, low_res):
    H,W = img_hr.size    
    ratio = random.choice(range(*low_res)) / min(H,W) 
    img_lr = img_hr.resize((round( H * ratio), round( W *ratio)))
    return img_lr

def create_blur_motion(img_hr, motion_blur, motion_blur_angle):
    blur_strength = random.choice(range(*motion_blur))
    blur_angle = random.choice(range(*motion_blur_angle))
    img_lr = apply_motion_blur(np.array(img_hr), blur_strength, blur_angle)
    img_lr = Image.fromarray( img_lr )
    return img_lr    

def create_g_blur(img_hr, g_blur):
    blur_strength = random.choice(range(*g_blur, 2)) + 1
    img_lr = transforms.GaussianBlur(blur_strength)(img_hr)
    return img_lr    