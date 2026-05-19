import logging
import os
import time
import torch
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval,R1_mAP_eval_LTCC, R1_mAP_eval_LaST
from torch.cuda import amp
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
import datetime
import numpy as np
from processor.eval_mevid import test_mevid
from processor.eval_ccvid import test as vid_test
from processor.train_fn import *
from collections import defaultdict
import pickle
import sys 
from torchvision.utils import save_image 

def visible_metric_items(metric_dict):
    return [(key, value) for key, value in metric_dict.items() if not str(key).startswith("__")]


def save_stage2_metric_bests(cfg, model, epoch, logger, rank1, mAP, cmc_overall, best_rank1_dict):
    if not cfg.TRAIN.TRAIN_VIDEO or not cfg.MODEL.DIST_TRAIN or dist.get_rank() != 0:
        return

    metrics = {
        "map": float(mAP),
    }
    overall_cmc = cmc_overall.get("__cmc_overall") if isinstance(cmc_overall, dict) else None
    if overall_cmc is not None and len(overall_cmc) >= 10:
        metrics["r5"] = float(overall_cmc[4])
        metrics["r10"] = float(overall_cmc[9])

    for name, value in metrics.items():
        best_key = f"__stage2_best_{name}"
        if value > best_rank1_dict[best_key]:
            best_rank1_dict[best_key] = value
            path = os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + f"_best_{name}.pth")
            logger.info(f"Saving stage2 best {name.upper()} checkpoint at epoch {epoch}: {value:.4f}")
            torch.save(model.state_dict(), path)


def default_img_loader(cfg, data, ):
    text = None
    jepa_paths = None
    if cfg.MODEL.ADD_META:
        if cfg.MODEL.JEPA_ENABLE and len(data) == 7:
            samples, targets, camids, _,clothes, meta, jepa_paths = data
        else:
            samples, targets, camids, _,clothes, meta = data
        meta = [m.float() for m in meta]
        meta = torch.stack(meta, dim=0)
        meta = meta.cuda(non_blocking=True)
        if cfg.MODEL.MASK_META:
            meta[:, 5:21] = 0
            meta[:, 35:57] = 0
            meta[:, 84] = 0
            meta[:, 90] = 0
            meta[:, 92:96] = 0
            meta[:, 97] = 0
            meta[:, 100] = 0
            meta[:, 102:105] = 0
    else:
        if cfg.MODEL.JEPA_ENABLE and len(data) == 8:
            samples, targets, camids,_, clothes,meta,text, jepa_paths = data
        elif cfg.MODEL.JEPA_ENABLE and len(data) == 7:
            samples, targets, camids,_, clothes,meta, jepa_paths = data
        else:
            samples, targets, camids,_, clothes,meta,text = data
        meta = None

    samples = samples.cuda(non_blocking=True)
    targets = targets.cuda(non_blocking=True)
    clothes = clothes.cuda(non_blocking=True)
    
    return samples, targets, clothes, meta, camids, jepa_paths


def maybe_load_jepa_tokens(cfg, jepa_paths, device, training_mode):
    if not cfg.MODEL.JEPA_ENABLE:
        return None
    if training_mode not in ("image", "video"):
        return None
    if jepa_paths is None:
        raise RuntimeError("MODEL.JEPA_ENABLE is True, but the dataloader did not return JEPA frame paths.")
    from tools.jepa_lazy_cache import load_jepa_batch
    write_cache = cfg.MODEL.JEPA_WRITE_TRAIN_CACHE
    if training_mode == "image":
        write_cache = cfg.MODEL.JEPA_WRITE_IMAGE_CACHE
    return load_jepa_batch(cfg, jepa_paths, device, write_cache=write_cache)

def evaluate_fn(cfg, model, val_loader, logger, evaluator_diff, evaluator_same, val_loader_same, device, epoch, rank_writer, mAP_writer, dataset, eval_mode=None, evaluator=None, dump=None, evaluator_general=None, queryloader=None, galleryloader=None , aux_dump=None):
    model.eval()
    cmc_overall, mAP_overall = None, None 
    rank1, mAP, cmc_overall, mAP_overall = None, None, None, None 
    do_dump = (eval_mode) or (dump)
    dump_w_index=eval_mode
    if 'prcc' in cfg.DATA.DATASET:
        evaluator_diff.reset()
        evaluator_same.reset()
        logger.info("Clothes changing setting")
        if dump_w_index:
            if aux_dump:
                rank1, mAP= test_w_index_w_aux(cfg, model, evaluator_diff, val_loader, logger, device,epoch,rank_writer, mAP_writer, dump=do_dump, dataset_name=cfg.DATA.DATASET, prefix=" CC : ")
                logger.info("Standard setting")
                test_w_index_w_aux(cfg, model, evaluator_same, val_loader_same, logger, device,epoch,rank_writer, mAP_writer, test=True, dump=False, dataset_name=cfg.DATA.DATASET)
            else:
                rank1, mAP= test_w_index(cfg, model, evaluator_diff, val_loader, logger, device,epoch,rank_writer, mAP_writer, dump=do_dump, dataset_name=cfg.DATA.DATASET, prefix=" CC : ")
                logger.info("Standard setting")
                test_w_index(cfg, model, evaluator_same, val_loader_same, logger, device,epoch,rank_writer, mAP_writer, test=True, dump=False, dataset_name=cfg.DATA.DATASET)
        else:
            rank1, mAP= test(cfg, model, evaluator_diff, val_loader, logger, device,epoch, rank_writer, mAP_writer, prefix=" CC : ")
            logger.info("Standard setting")
            test(cfg, model, evaluator_same, val_loader_same, logger, device, epoch,  rank_writer, mAP_writer,test=True)
    elif 'ltcc' in cfg.DATA.DATASET:
        evaluator_diff.reset()
        evaluator_general.reset()

        if dump_w_index:
            logger.info("Clothes changing setting")
            if aux_dump:
                rank1, mAP = test_w_index_w_aux(cfg, model, evaluator_diff, val_loader, logger, device,epoch,rank_writer, mAP_writer, cc=True, dump=do_dump, dataset_name=cfg.DATA.DATASET)
                logger.info("Standard setting")
                test_w_index_w_aux(cfg, model, evaluator_general, val_loader, logger, device,epoch,rank_writer, mAP_writer, test=True, prefix=" General: ", dump=False, dataset_name=cfg.DATA.DATASET)
            else:
                rank1, mAP = test_w_index(cfg, model, evaluator_diff, val_loader, logger, device,epoch,rank_writer, mAP_writer, cc=True, dump=do_dump, dataset_name=cfg.DATA.DATASET)
                logger.info("Standard setting")
                test_w_index(cfg, model, evaluator_general, val_loader, logger, device,epoch,rank_writer, mAP_writer, test=True, prefix=" General: ", dump=False, dataset_name=cfg.DATA.DATASET)
        else:
            logger.info("Clothes changing setting")
            rank1, mAP = test(cfg, model, evaluator_diff, val_loader, logger, device,epoch,rank_writer, mAP_writer, cc=True)
            logger.info("Standard setting")
            test(cfg, model, evaluator_general, val_loader, logger, device, epoch, rank_writer, mAP_writer,test=True, prefix=" General: ")
    elif 'mevid' in cfg.DATA.DATASET :
        rank1, mAP, cmc_overall, mAP_overall = test_mevid(cfg, model, val_loader[0], val_loader[1], dataset, device, dump_w_index=eval_mode, dump=do_dump)
    elif 'ccvid' in cfg.DATA.DATASET :
        rank1, mAP = vid_test(cfg, model, val_loader[0], val_loader[1], dataset, device, dump_w_index=eval_mode, dump=do_dump)
    if 'mevid' in cfg.DATA.DATASET: 
        if dist.get_rank() == 0:
            st = "\n"
            for k1, k2 in zip(cmc_overall, mAP_overall): st += f" {k1} : {cmc_overall[k1]:.1%} & {k2} : {mAP_overall[k2]:.1%} \n"
            logger.info(f"==> {st}")
        dist.barrier()
    return rank1, mAP, cmc_overall, mAP_overall
        
def set_train_methods(cfg, model, local_rank):
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("EVA-attribure.train")
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],find_unused_parameters=True)

    if cfg.TENSORBOARD:
        train_writer = SummaryWriter(os.path.join(cfg.OUTPUT_DIR, 'train'))
        rank_writer = SummaryWriter(os.path.join(cfg.OUTPUT_DIR, 'rank'))
        mAP_writer = SummaryWriter(os.path.join(cfg.OUTPUT_DIR, 'mAP'))
    else:
        train_writer = None
        rank_writer = None 
        mAP_writer = None
        
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    
    scaler = amp.GradScaler()
    
    return eval_period, device, epochs, logger, train_writer, rank_writer, mAP_writer, loss_meter, acc_meter, scaler, model
    
def train_step(cfg, model, train_loader, optimizer, optimizer_center, loss_fn, scaler, loss_meter, acc_meter, scheduler, epoch, DEFAULT_LOADER=default_img_loader, train_writer=None , training_mode="image", **kwargs):
    log_period = cfg.SOLVER.LOG_PERIOD
    logger = logging.getLogger("EVA-attribure.train")

    model.train()
    for idx, data in enumerate(train_loader):

        samples, targets, clothes, meta, camids, text = DEFAULT_LOADER(cfg, data, )
        jepa_tokens = maybe_load_jepa_tokens(cfg, text, samples.device, training_mode)
        # save_image (normalize(samples), "temp.png")
        optimizer.zero_grad()
        optimizer_center.zero_grad()
        with amp.autocast(enabled=True):
            if cfg.MODEL.ADD_META:
                if cfg.MODEL.CLOTH_ONLY:
                    score, feat = model(samples, clothes, jepa_tokens=jepa_tokens)
                else:
                    score, feat = model(samples, clothes, meta)
            else:
                score, feat = model(samples)
        loss = loss_fn(score, feat, targets, camids, training_mode=training_mode)
        if cfg.TENSORBOARD:
            train_writer.add_scalar('loss', loss.item(), epoch)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
            for param in center_criterion.parameters():
                param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
            scaler.step(optimizer_center)
            scaler.update()
        if isinstance(score, list):
            acc = (score[0].max(1)[1] == targets).float().mean()
        else:
            acc = (score.max(1)[1] == targets).float().mean()

        loss_meter.update(loss.item(), samples.shape[0])
        acc_meter.update(acc, 1)

        torch.cuda.synchronize()
        if (idx + 1) % log_period == 0:
            logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                        .format(epoch, (idx + 1), len(train_loader),
                                loss_meter.avg, acc_meter.avg, scheduler._get_lr(epoch)[0]))
    return idx 
    
def eval_step(cfg, model, val_loader, evaluator_diff, evaluator_same, val_loader_same, device, epoch, rank_writer, mAP_writer, dataset, 
    best_rank1_dict, best_map_dict, best_rank1, best_map, evaluator=None, evaluator_general=None , queryloader=None, galleryloader=None, threshold_drop=10):
    logger = logging.getLogger("EVA-attribure.train")
    rank1, mAP, cmc_overall, mAP_overall =  evaluate_fn(cfg, model, val_loader, logger, evaluator_diff, evaluator_same, val_loader_same, device, epoch, rank_writer, mAP_writer, dataset, evaluator=evaluator, evaluator_general=evaluator_general, queryloader=queryloader, galleryloader=galleryloader)    
    best_epoch = 0 

    if 'mevid' in cfg.DATA.DATASET :
        for key, value in visible_metric_items(cmc_overall): best_rank1_dict[key]= max(best_rank1_dict[key], value)
        for key, value in visible_metric_items(mAP_overall): best_map_dict[key] = max(value, best_map_dict[key])
        
        rank1 = torch.tensor(rank1).cuda()
        dist.barrier()
        dist.broadcast(rank1, src=0)
        rank1 = rank1.cpu().item()
        
        
        
    # print(dist.get_rank(), rank1, best_rank1)
    if (best_rank1 < 1 and rank1 + threshold_drop / 100  < best_rank1) or (rank1 + threshold_drop < best_rank1):
        logger.info(f" \n\n\n *** TERMINATING... Top1 : {rank1:.1%} << Best Rank-1 {best_rank1:.1%} - {threshold_drop} *** \n\n\n")
        logger.info("==> Best Rank-1 {:.1%}, Best Map {:.1%} achieved at epoch {}".format(best_rank1, best_map, best_epoch))
        dist.barrier()
        sys.exit()

    is_best = rank1 > best_rank1
    best_map = max(best_map, mAP)
    save_stage2_metric_bests(cfg, model, epoch, logger, rank1, mAP, cmc_overall, best_rank1_dict)
    
    if is_best:
        best_rank1 = rank1
        best_epoch = epoch
        if dist.get_rank() == 0:
            logger.info("==> Best Rank-1 {:.1%}, Best Map {:.1%} achieved at epoch {}".format(best_rank1, best_map, best_epoch))
            if 'mevid' in cfg.DATA.DATASET :
                st = "\n"
                for (k1, v1), (k2, v2) in zip(visible_metric_items(best_rank1_dict), visible_metric_items(best_map_dict)): st += f" {k1} : {v1:.1%} & {k2} : {v2:.1%} \n"
                logger.info(f"==> {st}")
        if cfg.MODEL.DIST_TRAIN:
            if dist.get_rank() == 0:
                logger.info(f"Saving the model Now ....., {rank1:.4f}, {mAP:.4f}")
                torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_best.pth'))

    if cfg.MODEL.DIST_TRAIN and dist.get_rank() == 0 and 'briar' in cfg.DATA.DATASET :
        torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + f'_{epoch}.pth'))

    return best_map, best_rank1, best_epoch, best_rank1_dict, best_map_dict 

def evaluator_gen(cfg, dataset):
    evaluator_diff, evaluator_general, evaluator_same, evaluator = None, None, None, None 
    if 'ltcc' in cfg.DATA.DATASET :
        evaluator_diff = R1_mAP_eval_LTCC(dataset.num_query_imgs, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, length = len(dataset.query) + len(dataset.gallery))  # ltcc
        evaluator_general = R1_mAP_eval(dataset.num_query_imgs, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, length = len(dataset.query) + len(dataset.gallery))
        evaluator_diff.reset()
        evaluator_general.reset()
    elif 'prcc' in cfg.DATA.DATASET :
        evaluator_diff = R1_mAP_eval(dataset.num_query_imgs_diff, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, length = dataset.num_query_imgs_diff + len(dataset.gallery))  # prcc
        evaluator_same = R1_mAP_eval(dataset.num_query_imgs_same, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, length = dataset.num_query_imgs_same + len(dataset.gallery))
        evaluator_diff.reset()
        evaluator_same.reset()
    elif ('mevid' in cfg.DATA.DATASET) or ('casia' in cfg.DATA.DATASET) or ('ccvid' in cfg.DATA.DATASET) or ('briar' in cfg.DATA.DATASET):
        evaluator_diff, evaluator_general, evaluator_same = None, None, None
    elif "last" in cfg.DATA.DATASET:
        evaluator = R1_mAP_eval_LaST(num_query=dataset.num_query_imgs, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, q_length = len(dataset.query) , g_length = len(dataset.gallery))
        evaluator.reset()
    else:
        evaluator = R1_mAP_eval(dataset.num_query_imgs, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, length = len(dataset.query) + len(dataset.gallery))
        evaluator.reset()
    return evaluator_diff, evaluator_general, evaluator_same, evaluator



################ Train ################
#### Default Normal Video / Image 
def do_train(cfg,
             model,
             center_criterion,
             train_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             local_rank,
             dataset,
             val_loader = None,
             val_loader_same = None,
             eval=None, save5=None, TRAIN_step_FN=train_step, training_mode="image", queryloader=None, galleryloader=None, threshold_drop=10, **kwargs):

    eval_period, device, epochs, logger, train_writer, rank_writer, \
        mAP_writer, loss_meter, acc_meter, scaler, model = set_train_methods(cfg, model, local_rank)
    
    evaluator_diff, evaluator_general, evaluator_same, evaluator = evaluator_gen(cfg, dataset)

    best_rank1 = -np.inf
    best_map = -np.inf
    best_epoch = 0
    mAP = 0 
    best_map_dict = defaultdict(int)
    best_rank1_dict = defaultdict(int)
    start_train_time = time.time()
    
    if eval:
        if cfg.TRAIN_DUMP:
            if TRAIN_step_FN == train_w_color_labels:
                train_w_color_labels_dump(cfg, model, train_loader, optimizer, optimizer_center, loss_fn, scaler, loss_meter, acc_meter, scheduler,  -1, train_writer=train_writer, training_mode=training_mode, **kwargs )
            elif TRAIN_step_FN == train_step:
                train_step_labels_dump(cfg, model, train_loader, optimizer, optimizer_center, loss_fn, scaler, loss_meter, acc_meter, scheduler,  -1, train_writer=train_writer, training_mode=training_mode, **kwargs )
            else:
                assert False, "not verified yet"
        elif cfg.AUX_DUMP:
            rank1, mAP, _, _ =  evaluate_fn(cfg, model, val_loader, logger, evaluator_diff, evaluator_same, val_loader_same, device, -1, rank_writer, mAP_writer, dataset, eval_mode=cfg.TEST.MODE, evaluator=evaluator, dump=True, evaluator_general=evaluator_general, queryloader=queryloader, galleryloader=galleryloader, aux_dump=True)    
            logger.info("==> EVAL: Rank-1 {:.1%}, Map {:.1%} achieved".format(rank1, mAP))
        else:
            rank1, mAP, _, _ =  evaluate_fn(cfg, model, val_loader, logger, evaluator_diff, evaluator_same, val_loader_same, device, -1, rank_writer, mAP_writer, dataset, eval_mode=cfg.TEST.MODE, evaluator=evaluator, dump=True, evaluator_general=evaluator_general, queryloader=queryloader, galleryloader=galleryloader,)    
            logger.info("==> EVAL: Rank-1 {:.1%}, Map {:.1%} achieved".format(rank1, mAP))
        return 
    idx =0            
    logger.info('start training')
    logger.info("Train Start !!")
    for epoch in range(cfg.TRAIN.START_EPOCH, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()

        scheduler.step(epoch)
        idx = TRAIN_step_FN(cfg, model, train_loader, optimizer, optimizer_center, loss_fn, scaler, loss_meter, acc_meter, scheduler,  epoch, train_writer=train_writer, training_mode=training_mode, **kwargs )

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (idx + 1)

        if epoch % eval_period == 0:
            best_map, best_rank1, potential_best_epoch, best_rank1_dict, best_map_dict = eval_step(cfg, model, val_loader, evaluator_diff, evaluator_same, val_loader_same, device, epoch, rank_writer, mAP_writer, dataset,
            best_rank1_dict, best_map_dict, best_rank1, best_map, evaluator=evaluator, evaluator_general=evaluator_general, queryloader=queryloader, galleryloader=galleryloader, threshold_drop=threshold_drop)
            best_epoch = max(best_epoch, potential_best_epoch)
        
        if save5 and epoch % 5 == 0 and dist.get_rank() == 0:
            torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + f'_{epoch}.pth'))

            
    total_time = time.time() - start_train_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))
    logger.info("==> Best Rank-1 {:.1%}, Best Map {:.1%} achieved at epoch {}".format(best_rank1, best_map, best_epoch))
    if 'mevid' in cfg.DATA.DATASET :
        st = "\n"
        for (k1, v1), (k2, v2) in zip(visible_metric_items(best_rank1_dict), visible_metric_items(best_map_dict)): st += f" {k1} : {v1:.1%} & {k2} : {v2:.1%} \n"
        logger.info(f"==> {st}")

    if dist.get_rank() == 0:
        torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_last.pth'))
  

def do_train_w_teachers(cfg,
             model,
             center_criterion,
             train_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             local_rank,
             dataset,
             val_loader = None,
             val_loader_same = None,
             eval=None, save5=None, TRAIN_step_FN=train_step, TRAIN_ext_step_FN=train_step,
             teacher_trainloader=None, teacher_dataset=None, training_mode="image", teacher_training_mode="video", queryloader=None, galleryloader=None, threshold_drop=10, **kwargs):

    eval_period, device, epochs, logger, train_writer, rank_writer, \
        mAP_writer, loss_meter, acc_meter, scaler, model = set_train_methods(cfg, model, local_rank)
    
    evaluator_diff, evaluator_general, evaluator_same, evaluator = evaluator_gen(cfg, dataset)

    loss_meter_teacher = AverageMeter()
    acc_meter_teacher = AverageMeter()
    best_rank1 = -np.inf
    best_map = -np.inf
    best_epoch = 0
    mAP = 0 
    best_map_dict = defaultdict(int)
    best_rank1_dict = defaultdict(int)
    start_train_time = time.time()
    
    if eval:
        rank1, mAP, _, _ =  evaluate_fn(cfg, model, val_loader, logger, evaluator_diff, evaluator_same, val_loader_same, device, -1, rank_writer, mAP_writer, dataset, eval_mode=cfg.TEST.MODE, evaluator=evaluator, dump=True, evaluator_general=evaluator_general, queryloader=queryloader, galleryloader=galleryloader)    
        logger.info("==> EVAL: Rank-1 {:.1%}, Map {:.1%} achieved".format(rank1, mAP))
        return 
    idx =0            
    DEFAULT_LOADER=default_img_loader
    logger.info('start training')
    logger.info("Train Start !!")
    
    for epoch in range(cfg.TRAIN.START_EPOCH, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        loss_meter_teacher.reset()

        acc_meter.reset()
        acc_meter_teacher.reset()

        scheduler.step(epoch)
        if not cfg.TRAIN.DEBUG:
            logger.info("==> Teacher Training .... ")
            model.module.student_mode = True
            model.student_mode = True 
            TRAIN_ext_step_FN(cfg, model, teacher_trainloader, optimizer,  optimizer_center,  loss_fn,  scaler, loss_meter_teacher, acc_meter_teacher, scheduler,   epoch, train_writer=train_writer, training_mode=teacher_training_mode, **kwargs )    
                            # cfg, model, train_loader,        optimizer,  optimizer_center,   loss_fn, scaler, loss_meter,         acc_meter,         scheduler,   epoch, train_writer=train_writer, training_mode=training_mode,         **kwargs )
            model.student_mode = False 
            model.module.student_mode = False

        logger.info("==> Student Training .... ")
        idx = TRAIN_step_FN(cfg, model, train_loader, optimizer, optimizer_center, loss_fn, scaler, loss_meter, acc_meter, scheduler,  epoch, train_writer=train_writer, training_mode=training_mode, **kwargs )
        end_time = time.time()
        time_per_batch = (end_time - start_time) / (idx + 1)

        # name = os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + f'_{epoch}.pth')
        # torch.save(model.state_dict(), name)

        if epoch % eval_period == 0:
            best_map, best_rank1, potential_best_epoch, best_rank1_dict, best_map_dict = eval_step(cfg, model, val_loader, evaluator_diff, evaluator_same, val_loader_same, device, epoch, rank_writer, mAP_writer, dataset,
            best_rank1_dict, best_map_dict, best_rank1, best_map, evaluator=evaluator, evaluator_general=evaluator_general, queryloader=queryloader, galleryloader=galleryloader, threshold_drop=threshold_drop)
            best_epoch = max(best_epoch, potential_best_epoch)
        
        if save5 and epoch % 5 == 0 and dist.get_rank() == 0:
            torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + f'_{epoch}.pth'))

    total_time = time.time() - start_train_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))
    logger.info("==> Best Rank-1 {:.1%}, Best Map {:.1%} achieved at epoch {}".format(best_rank1, best_map, best_epoch))
    if 'mevid' in cfg.DATA.DATASET :
        st = "\n"
        for (k1, v1), (k2, v2) in zip(visible_metric_items(best_rank1_dict), visible_metric_items(best_map_dict)): st += f" {k1} : {v1:.1%} & {k2} : {v2:.1%} \n"
        logger.info(f"==> {st}")
                
    if dist.get_rank() == 0:
        torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_last.pth'))
  


################ Eval ################
def do_inference(cfg,
                 model,
                 dataset,
                 val_loader = None,
                 val_loader_same=None):
    logger = logging.getLogger("EVA-attribure.test")
    logger.info("Enter inferencing")

    logger.info("transreid inferencing")
    device = "cuda"
    if 'ltcc' in cfg.DATA.DATASET:
        evaluator_diff = R1_mAP_eval_LTCC(dataset.num_query_imgs, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)  # ltcc
        evaluator_general = R1_mAP_eval(dataset.num_query_imgs, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    elif 'prcc' in cfg.DATA.DATASET:
        evaluator_diff = R1_mAP_eval(dataset.num_query_imgs_diff, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)  # prcc
        evaluator_same = R1_mAP_eval(dataset.num_query_imgs_same, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    else:
        evaluator = R1_mAP_eval(dataset.num_query_imgs, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    if 'ltcc' in cfg.DATA.DATASET:
        evaluator_diff.reset()
        evaluator_general.reset()

    elif 'prcc' in cfg.DATA.DATASET:
        evaluator_diff.reset()
        evaluator_same.reset()
    else:
        evaluator.reset()
    model.to(device)
    model.eval()
    if 'prcc' in cfg.DATA.DATASET:
        logger.info("Clothes changing setting")
        test(cfg, model, evaluator_diff, val_loader, logger, device, test=True)
        logger.info("Standard setting")
        test(cfg, model, evaluator_same, val_loader_same, logger, device, test=True)
    elif 'ltcc' in cfg.DATA.DATASET :
        logger.info("Clothes changing setting")
        test(cfg, model, evaluator_diff, val_loader, logger, device, test=True,cc=True)
        logger.info("Standard setting")
        test(cfg, model, evaluator_general, val_loader, logger, device, test=True)
    else:
        test(cfg, model, evaluator, val_loader, logger, device, test=True)

def test(cfg, model, evaluator, val_loader, logger, device, epoch=None, rank_writer=None, mAP_writer=None,test=False,cc=False, prefix=None, dump=None):
    for n_iter, (imgs, pids, camids, clothes_id, clothes_ids, meta) in enumerate(val_loader):
        with torch.no_grad():
            imgs = imgs.to(device)
            # save_image (normalize(imgs), "temp.png")
            meta = meta.to(device)
            clothes_ids = clothes_ids.to(device)
            meta = meta.to(torch.float32)
            if cfg.MODEL.CLOTH_ONLY:
                feat = model(imgs, clothes_ids)
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
            if cc:
                evaluator.update((feat, pids, camids, clothes_id))
            else:
                evaluator.update((feat, pids, camids))
    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    
    if not prefix:
        prefix = " CC: " if cc else  " SC: "
    string = "{} CMC curve, ".format(prefix)
    for r in [1, 5, 10]:
        string+= "Rank-{:<3}:{:.1%}  ".format(r, cmc[r - 1])
    logger.info(string)
    logger.info("{} mAP Acc. :{:.1%}".format(prefix,  mAP ))
    if test :
        torch.cuda.empty_cache()
        return
    logger.info("Validation Results - Epoch: {}".format(epoch))
    rank1 = cmc[0]
    if rank_writer:
        rank_writer.add_scalar('rank1', rank1, epoch)
        mAP_writer.add_scalar('mAP', mAP, epoch)
    torch.cuda.empty_cache()
    return rank1, mAP



def test_w_index(cfg, model, evaluator, val_loader, logger, device, epoch=None, rank_writer=None, mAP_writer=None,test=False,cc=False, prefix=None, dump=None, dataset_name=None):
    indices = []
    for n_iter, (imgs, pids, camids, clothes_id, clothes_ids, meta, index) in enumerate(val_loader):
        with torch.no_grad():
            imgs = imgs.to(device)
            meta = meta.to(device)
            clothes_ids = clothes_ids.to(device)
            meta = meta.to(torch.float32)
            if cfg.MODEL.CLOTH_ONLY:
                feat = model(imgs, clothes_ids)
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
            if cc:
                evaluator.update((feat, pids, camids, clothes_id))
            else:
                evaluator.update((feat, pids, camids))
            indices.append(index)
    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    
    if not prefix:
        prefix = " CC: " if cc else  " SC: "
    string = "{} CMC curve, ".format(prefix)
    for r in [1, 5, 10]:
        string+= "Rank-{:<3}:{:.1%}  ".format(r, cmc[r - 1])
    logger.info(string)
    logger.info("{} mAP Acc. :{:.1%}".format(prefix,  mAP ))
    if test :
        torch.cuda.empty_cache()
        return
    logger.info("Validation Results - Epoch: {}".format(epoch))
    rank1 = cmc[0]
    if rank_writer:
        rank_writer.add_scalar('rank1', rank1, epoch)
        mAP_writer.add_scalar('mAP', mAP, epoch)
    torch.cuda.empty_cache()

    if dump:
        if dist.get_rank() == 0:
            indices = torch.cat(indices)
            if cfg.TAG:
                evaluator.dump_vals(indices, cfg.TAG)
            else:
                evaluator.dump_vals(indices, dataset_name)
        dist.barrier()

    return rank1, mAP

def test_w_index_w_aux(cfg, model, evaluator, val_loader, logger, device, epoch=None, rank_writer=None, mAP_writer=None,test=False,cc=False, prefix=None, dump=None, dataset_name=None):
    print(" *** DUMPING WITH 2 AUX OUTPUT *** ")
    indices = []
    model.dump_aux = True 
    
    extrafeats = []
    extrafeats_outputs = [] 
    for n_iter, (imgs, pids, camids, clothes_id, clothes_ids, meta, index) in enumerate(val_loader):
        with torch.no_grad():
            imgs = imgs.to(device)
            meta = meta.to(device)
            clothes_ids = clothes_ids.to(device)
            meta = meta.to(torch.float32)
            feat, extra_token_feats, extra_token_output = model(imgs, clothes_ids)
            extrafeats_outputs.append(extra_token_output)
            extrafeats.append(extra_token_feats)
            if cfg.TEST.CONCAT_COLORS:
                feat = torch.cat([feat, extra_token_feats],-1)
            if cc:
                evaluator.update((feat, pids, camids, clothes_id))
            else:
                evaluator.update((feat, pids, camids))
            indices.append(index)
    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    
    if not prefix:
        prefix = " CC: " if cc else  " SC: "
    string = "{} CMC curve, ".format(prefix)
    for r in [1, 5, 10]:
        string+= "Rank-{:<3}:{:.1%}  ".format(r, cmc[r - 1])
    logger.info(string)
    logger.info("{} mAP Acc. :{:.1%}".format(prefix,  mAP ))
    if test :
        torch.cuda.empty_cache()
        return
    logger.info("Validation Results - Epoch: {}".format(epoch))
    rank1 = cmc[0]
    if rank_writer:
        rank_writer.add_scalar('rank1', rank1, epoch)
        mAP_writer.add_scalar('mAP', mAP, epoch)
    torch.cuda.empty_cache()

    if dump:
        if dist.get_rank() == 0:
            from utils.metrics import concat_all_gather
            extrafeats = torch.cat(extrafeats)
            extrafeats_outputs = torch.cat(extrafeats_outputs)

            extrafeats, extrafeats_outputs = concat_all_gather([extrafeats, extrafeats_outputs], evaluator.length)

            extrafeats_qf = extrafeats[:evaluator.num_query]
            extrafeats_outputs_qf = extrafeats_outputs[:evaluator.num_query]

            extrafeats_gf = extrafeats[evaluator.num_query:]
            extrafeats_outputs_gf = extrafeats_outputs[evaluator.num_query:]
            
            indices = torch.cat(indices)
            if cfg.TAG:
                evaluator.dump_vals(indices, cfg.TAG, aux_dump=dict(extrafeats_qf=extrafeats_qf, extrafeats_outputs_qf=extrafeats_outputs_qf, extrafeats_gf=extrafeats_gf, extrafeats_outputs_gf=extrafeats_outputs_gf))
            else:
                evaluator.dump_vals(indices, dataset_name, aux_dump=dict(extrafeats_qf=extrafeats_qf, extrafeats_outputs_qf=extrafeats_outputs_qf, extrafeats_gf=extrafeats_gf, extrafeats_outputs_gf=extrafeats_outputs_gf))
        dist.barrier()
    model.dump_aux = False 
    return rank1, mAP
