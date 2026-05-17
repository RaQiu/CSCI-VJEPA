import os
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO" 
from torch import distributed as dist

import numpy as np
import torch
import random
from config import cfg
import argparse
from data import build_dataloader
from utils.logger import setup_logger
from model import build_model
from utils import auto_resume_helper,load_checkpoint


from train import set_seed, modify_params, set_up_params, set_up_dist, setup_logging, \
    setup_model

from processor import * 

if __name__ == "__main__":
    args, cfg = set_up_params()
    cfg = modify_params(cfg, args)
    local_rank, dist_local_rank, output_dir = set_up_dist(cfg, args)
    logger = setup_logging(cfg, args, output_dir, dist_local_rank)

    TRAIN_FN= do_train
    if "prcc" in cfg.DATA.DATASET:
        _, queryloader_same, queryloader_diff, galleryloader, dataset, _, val_loader, val_loader_same= build_dataloader(cfg, local_rank=args.local_rank)  # prcc_test
    else:
        _, queryloader, galleryloader, dataset, _, val_loader = build_dataloader(cfg, local_rank=args.local_rank)

    model, _, _, _, _, _ = setup_model(cfg, args, logger, dataset, )

    model.load_param(cfg.TEST.WEIGHT)
    if 'prcc' in cfg.DATA.DATASET:
        TRAIN_FN(
            cfg, model, None,
            None, None, None,
            None, None, args.local_rank, dataset,
            val_loader=val_loader, val_loader_same=val_loader_same,  eval=args.eval)
    else:
        TRAIN_FN(
            cfg, model, None, None, None, None, None,
            None, args.local_rank, dataset,
            val_loader=val_loader, eval=args.eval)
