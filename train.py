from utils.logger import setup_logger
from data import build_dataloader
from solver import make_optimizer
from solver.scheduler_factory import create_scheduler
from loss import make_loss
from processor import * 
from processor.train_fn import * 
import random
import torch
import numpy as np
import os
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO" 
from torch import distributed as dist
import argparse
from config import cfg
from model import build_model
from loss.custom_loss import KL_Loss, MSE
# from utilss import  
from utils import auto_resume_helper,load_checkpoint
from loss.custom_loss import * 

from data import __factory

vid_set = {key for key in __factory.keys() if "mevid" in key or "ccvid" in key}


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def modify_params(cfg, args, dumping_root="Dump/"):
    cfg.defrost()
    cfg.OUTPUT_DIR = os.path.join(dumping_root, cfg.OUTPUT_DIR)
    if args.return_index:
        cfg.TEST.MODE = True
    if cfg.TRAIN.TRAIN_VIDEO and (not cfg.DATA.F8):    
        cfg.AUG.SEQ_LEN = 4 
    if args.cal_eval: 
        cfg.MODEL.EMBED_DIM = 2048 * 2
    if cfg.TRAIN.COLOR_PROFILE:
        color_profile = cfg.TRAIN.COLOR_PROFILE
        multiply_factor = 1
        if color_profile in [24, 25, 40, 41, 49, 23, 32, 28, 48, 29, 27, 26]:multiply_factor= 3
        if color_profile in [50, 51, 52, 53, 54, 55, 56, 57]:multiply_factor= 20
        
        dim = 32
        if color_profile in [42, 49, 47, 48, 44, 43]:dim = 64
        elif color_profile in [40, 41, 35, 39, 38, 34]:dim = 16
        elif color_profile in [50, 51, 52, 53, 54, 55, 56, 57]:dim = 20 
        cfg.MODEL.EXTRA_DIM = multiply_factor * dim * dim

    cfg.freeze()
    return cfg 

def set_up_params():
    parser = argparse.ArgumentParser(description="CC_ReID  Training")
    parser.add_argument("--config_file", default="", help="path to config file", type=str)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", "--local-rank", default=0, type=int)
    parser.add_argument('--multi-node', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--no-head', action='store_true')

    parser.add_argument('--cal_eval', action='store_true')
    parser.add_argument('--save5', action='store_true')
    parser.add_argument('--env', type=str, default="nccl")
    
    parser.add_argument('--return-index', action='store_true')
    
    args = parser.parse_args()
    if args.config_file != "":
        print(f" **** {args.config_file}  **** ")
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    return args, cfg

def set_up_dist(cfg, args, dumping_root="Dump/"):
    set_seed(cfg.SOLVER.SEED)
    if cfg.MODEL.DIST_TRAIN:
        print("******", args.local_rank)
        torch.cuda.set_device(args.local_rank)
    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    if args.multi_node:
        world_size = int(os.environ.get('WORLD_SIZE', os.environ.get('SLURM_NTASKS')))
        rank = int(os.environ.get('RANK', os.environ.get('SLURM_PROCID')))
        local_rank = int(os.environ.get('LOCAL_RANK', os.environ.get('SLURM_LOCALID')))
        print(rank , local_rank , world_size)    
        dist.init_process_group(args.env, world_size=world_size, rank=rank)
        dist_local_rank = dist.get_rank()
        dist_world_size = dist.get_world_size()
        print("=====", local_rank, dist_local_rank, dist_world_size)
    else:
        dist.init_process_group(backend=args.env, init_method='env://')
        dist_local_rank = dist.get_rank()
        local_rank = None 

    # if cfg.MODEL.DIST_TRAIN:
    #     torch.distributed.init_process_group(backend='nccl', init_method='env://')
    print("******", args.local_rank, local_rank, dist_local_rank)
    # ****** 1 None 1
    # ****** 0 None 0

    # ****** 0 0 0
    # ****** 0 0 3
    # ****** 0 0 1
    # ****** 0 0 2
    return local_rank, dist_local_rank, output_dir

def setup_logging(cfg, args, output_dir, dist_local_rank):
    logger = setup_logger("EVA-attribure", output_dir, if_train=True, local_rank=dist_local_rank)
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))
    return logger

def setup_model(cfg, args, logger, dataset, ):
    model = build_model(cfg, dataset.num_train_pids, dataset.num_train_clothes)
    if args.resume:
        logger.info(f"\n *** Resuming From : {cfg.TEST.WEIGHT} ***  \n ")
        load_head = (not args.no_head)
        model.load_param(cfg.TEST.WEIGHT, load_head=load_head)
    loss_func, center_criterion = make_loss(cfg, num_classes=dataset.num_train_pids)

    optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)
    scheduler = create_scheduler(cfg, optimizer)
    return model, loss_func, center_criterion, optimizer, optimizer_center, scheduler
    
def add_additional_attributes(cfg, args):
    TRAIN_FN= do_train
    kwargs = {}
    kwargs['threshold_drop'] = 10 
    if cfg.TRAIN.COLOR_ADV:
        if cfg.TRAIN.COLOR_ADV or ("_att" in cfg.DATA.DATASET):
            kwargs["TRAIN_step_FN"] = train_w_color_labels
            if cfg.MODEL.ATT_AS_INPUT:
                kwargs["TRAIN_step_FN"] = train_w_color_labels_feed
            if cfg.MODEL.ATT_DIRECT:
                kwargs["TRAIN_step_FN"] = train_w_color_direct    
        kwargs["color_loss_fn"] = MSE(mean=False)
        if ( cfg.TRAIN.COLOR_ADV or "_att" in cfg.DATA.DATASET):
            if cfg.TRAIN.COLOR_LOSS == "cosine":
                kwargs["color_loss_fn"] = Cosine_Similarity()    
            if cfg.TRAIN.COLOR_LOSS == "ce":
                kwargs["color_loss_fn"] = nn.CrossEntropyLoss()
            

        # kwargs["pair_loss"] = pair_loss
        kwargs["ce"] = F.cross_entropy
        kwargs['mse'] = MSE(mean=False)
        kwargs['cosine'] = Cosine_Similarity()
        kwargs['distentangle'] = Cosine_Disentangle()

    if cfg.TRAIN.COLOR_PROFILE == -1:
        kwargs["TRAIN_step_FN"] = train_w_cl_dist
        kwargs["color_loss_fn"] = F.cross_entropy

    if args.save5:
        kwargs["save5"] = True 

    if cfg.DATA.DATASET in vid_set:
        kwargs["training_mode"] = "video" 
        kwargs['threshold_drop'] = 8    
        # if "mevid" in cfg.DATA.DATASET:
        #     kwargs['threshold_drop'] = 5    
    else:
        kwargs["training_mode"] = "image" 
        kwargs['threshold_drop'] = 10 
        if "ltcc" in cfg.DATA.DATASET:
            kwargs['threshold_drop'] = 5    

    return TRAIN_FN, kwargs




def compute_flops(model, verbose=False, print_per_layer_stat=False, resolution =(3, 224, 224) ):
    from ptflops import get_model_complexity_info
    import re
    macs, params = get_model_complexity_info(model.float(),  resolution , as_strings=True, print_per_layer_stat=print_per_layer_stat, verbose=verbose)
    flops = eval(re.findall(r'([\d.]+)', macs)[0])*2
    flops_unit = re.findall(r'([A-Za-z]+)', macs)[0][0]
    print('Computational complexity: {:<8}'.format(macs))
    print('Computational complexity: {} {}Flops'.format(flops, flops_unit))
    print('Number of parameters: {:<8}'.format(params))    
    # quit()

def compute_all_stats(model, trainloader, optimizer, local_rank, loss_func):
    print(f" Model parameters: {np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
    # compute_flops(model, verbose=False, print_per_layer_stat=False, resolution =(3, 224, 224) )
    # compute_flops(model, verbose=False, print_per_layer_stat=True, resolution =(3, 224, 224) )
    
    from thop import profile
    from thop import clever_format
    input = torch.randn(1, 3, 224, 224)
    macs, params = profile(model, inputs=(input, ))
    flops, params = clever_format([macs, params], "%.8f")
    print(f" GFLOP USING `thop` {macs/10 ** 9:.2f} MACs(G) '# of Params using thop': {params}M")
    
    from fvcore.nn import FlopCountAnalysis
    flops = FlopCountAnalysis(model, input)
    print(f"FLOP TOTAL : {flops.total()}" )
    per_modules = flops.by_module()
    keyss = {key for key in per_modules if "block" in key and "attn" in key and per_modules[key] != 0 and "norm" not in key and 'proj' not in key}
    selected_per_module = {key: per_modules[key] for key in keyss}
    print(f"FLOP BY MODULES : {selected_per_module}" )
    # print(f"FLOP BY MODULES & OPERATOR : {flops.by_module_and_operator()}" )
    
    
    
    


if __name__ == '__main__':
    args, cfg = set_up_params()
    cfg = modify_params(cfg, args)
    local_rank, dist_local_rank, output_dir = set_up_dist(cfg, args)
    logger = setup_logging(cfg, args, output_dir, dist_local_rank)

    
    # os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    if "prcc" in cfg.DATA.DATASET:
        trainloader, queryloader_same, queryloader_diff, galleryloader, dataset, train_sampler,val_loader,val_loader_same= build_dataloader(cfg, local_rank=args.local_rank)  # prcc_test
    else:
        trainloader, queryloader, galleryloader, dataset, train_sampler,val_loader = build_dataloader(cfg, local_rank=args.local_rank)

    model, loss_func, center_criterion, optimizer, optimizer_center, scheduler = setup_model(cfg, args, logger, dataset, )
    TRAIN_FN, kwargs = add_additional_attributes(cfg, args)


    if cfg.ANALYSIS_STATS:
        compute_all_stats(model, trainloader, optimizer, args.local_rank, loss_func)
        quit() 
        
    if 'prcc' in cfg.DATA.DATASET:
        TRAIN_FN(
            cfg, model, center_criterion,
            trainloader, optimizer, optimizer_center,
            scheduler, loss_func, args.local_rank, dataset,
            val_loader=val_loader, val_loader_same=val_loader_same,  eval=args.eval, **kwargs
        )
    else:
        TRAIN_FN(
            cfg, model, center_criterion, trainloader, optimizer, optimizer_center, scheduler,
            loss_func, args.local_rank, dataset,
            val_loader=val_loader, eval=args.eval, queryloader=queryloader, galleryloader=galleryloader, **kwargs 
        )
