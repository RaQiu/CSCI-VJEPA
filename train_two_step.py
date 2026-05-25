from data import build_dataloader
from processor import do_train_w_teachers
import os
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO" 

from train import set_seed, set_up_params, set_up_dist, setup_logging, setup_model, modify_params, add_additional_attributes, vid_set
from processor.train_fn import *
from processor import *
from loss.custom_loss import *
import torch
import types
from solver import make_optimizer
from solver.scheduler_factory import create_scheduler


def _freeze_backbone_keep_jepa(cfg, model, center_criterion, logger):
    """
    冻结 CSCI-V 主干,只让 JEPA 旁路 (+ 可选 head/classifier) 可训。
    在 setup_model 之后、DDP wrap 之前调用。返回新的 optimizer / optimizer_center / scheduler。
    """
    keep_head = cfg.MODEL.JEPA_FREEZE_KEEP_HEAD
    trainable_prefixes = ('jepa_side.', 'jepa_id_refine.')
    if keep_head:
        # head: 视频分类头 (cls_score = self.head(feat_h)), 必须可训以适应 JEPA residual
        # head_image: 图像分类头 (image_forward 用), stage2 student 阶段也走
        # bottleneck / classifier / fc_norm: 兜底,EZ_Eva 家族常见命名
        trainable_prefixes = trainable_prefixes + (
            'head.', 'head_image.', 'bottleneck.', 'classifier.', 'fc_norm.',
        )

    n_total, n_trainable, n_frozen = 0, 0, 0
    trainable_top = set()
    for name, p in model.named_parameters():
        n_total += p.numel()
        if any(name.startswith(pref) for pref in trainable_prefixes):
            p.requires_grad = True
            n_trainable += p.numel()
            trainable_top.add(name.split('.', 1)[0])
        else:
            p.requires_grad = False
            n_frozen += p.numel()

    # 让冻结子模块保持 eval (关 dropout、BN running stats),即使 epoch 开头调用 model.train()
    frozen_modules = [
        m for m_name, m in model.named_children()
        if m_name not in trainable_top
    ]
    orig_train = model.train

    def _train_keep_frozen_eval(self, mode=True):
        orig_train(mode)
        for m in frozen_modules:
            m.eval()
        return self
    model.train = types.MethodType(_train_keep_frozen_eval, model)
    model.train(True)

    if logger is not None:
        logger.info(
            f"[JEPA_FREEZE_BACKBONE] trainable={n_trainable:,} / total={n_total:,} "
            f"({100.0*n_trainable/max(1,n_total):.2f}%), frozen={n_frozen:,}"
        )
        logger.info(f"[JEPA_FREEZE_BACKBONE] trainable top-level modules: {sorted(trainable_top)}")

    # 重建 optimizer / scheduler — make_optimizer 会自动跳过 requires_grad=False
    optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)
    scheduler = create_scheduler(cfg, optimizer)
    return optimizer, optimizer_center, scheduler

def add_external_training_fns(kwargs_external, kwargs_internal):
    kwargs = {}
    
    kwargs.update(kwargs_external)    
    kwargs["teacher_training_mode"] = kwargs_external['training_mode']
    kwargs['training_mode']         = kwargs_internal['training_mode']

    if 'TRAIN_step_FN' in kwargs:
        del kwargs["TRAIN_step_FN"]
    if 'TRAIN_step_FN' in kwargs_external:
        kwargs['TRAIN_ext_step_FN'] = kwargs_external['TRAIN_step_FN']   
    if 'TRAIN_step_FN' in kwargs_internal:
        kwargs["TRAIN_step_FN"] = kwargs_internal['TRAIN_step_FN']

    return kwargs

if __name__ == '__main__':

    args, cfg = set_up_params()
    cfg = modify_params(cfg, args)
    local_rank, dist_local_rank, output_dir = set_up_dist(cfg, args)
    logger = setup_logging(cfg, args, output_dir, dist_local_rank)
    
    ################  TEACHER MODE  ################
    ################  IMAGE PART OF VIDEO MODEL ################
    kwargs = {}
    
    student_dataset = cfg.DATA.DATASET
    student_dataset_root = cfg.DATA.ROOT
    student_model = cfg.MODEL.NAME
    student_resume = args.resume
    student_test_weight = cfg.TEST.WEIGHT
    student_dataset_fix = cfg.DATA.DATASET_FIX
    batch_size = cfg.DATA.BATCH_SIZE
    
    cfg.defrost()
    cfg.DATA.DATASET = cfg.TRAIN.TEACH1
    cfg.DATA.ROOT = cfg.TRAIN.DIR_TEACH1    
    
    cfg.TRAIN.TRAIN_VIDEO = cfg.DATA.DATASET in vid_set
    if cfg.TRAIN.TEACH1_LOAD_AS_IMG:
        cfg.TRAIN.TRAIN_VIDEO = False 
    
    
    cfg.DATA.DATASET_FIX = cfg.TRAIN.TEACH_DATASET_FIX
    if cfg.TRAIN.TEACH1_MODEL:
        cfg.MODEL.NAME = cfg.TRAIN.TEACH1_MODEL
        cfg.TEST.WEIGHT = cfg.TRAIN.TEACH1_MODEL_WT
        args.resume = True 
    

    if cfg.TRAIN.TEACH1:
        teacher_trainloader, _, _, teacher_dataset, _, _ = build_dataloader(cfg, local_rank=args.local_rank, teacher_mode=True )
        cfg.TRAIN.TEACH1_NUMCLASSES = teacher_dataset.num_train_pids
    else:
        teacher_trainloader, teacher_dataset = None , None 
        cfg.TRAIN.TEACH1_NUMCLASSES = None
    
    if cfg.TRAIN.TEACH1_MODEL:
        teacher_model, _, _, _, _, _ = setup_model(cfg, args, logger, teacher_dataset, )
        teacher_model.eval()
        teacher_model.to(args.local_rank)
        teacher_model = torch.nn.parallel.DistributedDataParallel(teacher_model, device_ids=[args.local_rank],find_unused_parameters=True)
        kwargs['teacher_model'] = teacher_model
        kwargs['mse'] = MSE()
        args.resume = student_resume
        cfg.TEST.WEIGHT = student_test_weight

    kwargs_external = {"training_mode" : None }
    if not args.eval:
        _, kwargs_external = add_additional_attributes(cfg, args)
    if cfg.TRAIN.TEACH1_LOAD_AS_IMG:
        kwargs_external['training_mode'] = "image"
    
    cfg.MODEL.NAME = student_model
    cfg.DATA.BATCH_SIZE = batch_size
    cfg.DATA.DATASET = student_dataset
    cfg.DATA.ROOT = student_dataset_root
    args.cal_eval = False 
    if cfg.TRAIN.HYBRID:
        cfg.DATA.DATASET_FIX = None
        cfg.TRAIN.COLOR_PROFILE = None
    else:
        cfg.DATA.DATASET_FIX = student_dataset_fix
    cfg.DATA.SAMPLING_PERCENTAGE = None 
    cfg.DATA.DATASET_SAMPLING_PERCENTAGE = None
    cfg.TRAIN.TRAIN_VIDEO = cfg.DATA.DATASET in vid_set
    cfg.TRAIN.COLOR_ADV = None 
    cfg.freeze()

    ################  STUDENT MODE  ################
    # os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    if "prcc" in cfg.DATA.DATASET:
        trainloader, queryloader_same, queryloader_diff, galleryloader, dataset, train_sampler,val_loader,val_loader_same= build_dataloader(cfg, local_rank=args.local_rank)  # prcc_test
    else:
        trainloader, queryloader, galleryloader, dataset, train_sampler,val_loader = build_dataloader(cfg, local_rank=args.local_rank)

    if cfg.TRAIN.TEACH1_NUMCLASSES == None:
        cfg.defrost()
        cfg.TRAIN.TEACH1_NUMCLASSES = dataset.num_train_pids
        cfg.freeze()


    model, loss_func, center_criterion, optimizer, optimizer_center, scheduler = setup_model(cfg, args, logger, dataset, )

    # 可选: 冻结 CSCI-V 主干, 只训 JEPA 旁路 (+ 可选 head)
    # 必须在 setup_model 之后 (load_param 已恢复权重) 但 DDP wrap 之前
    if cfg.MODEL.JEPA_FREEZE_BACKBONE:
        if model.jepa_side is None:
            raise RuntimeError(
                "JEPA_FREEZE_BACKBONE=True 但 model.jepa_side 未构建; "
                "需要 MODEL.JEPA_SIDE_PATH=True"
            )
        optimizer, optimizer_center, scheduler = _freeze_backbone_keep_jepa(
            cfg, model, center_criterion, logger
        )

    _, kwargs_internal = add_additional_attributes(cfg, args)
    
    kwargs = add_external_training_fns(kwargs_external, kwargs_internal)

    if 'prcc' in cfg.DATA.DATASET :
        do_train_w_teachers(
            cfg, model, center_criterion, trainloader, optimizer, optimizer_center, scheduler, 
            loss_func, args.local_rank, dataset, teacher_trainloader=teacher_trainloader, teacher_dataset=teacher_dataset, 
            val_loader=val_loader, val_loader_same=val_loader_same, eval=args.eval, **kwargs)
    else:
        do_train_w_teachers(
            cfg, model, center_criterion, trainloader, optimizer, optimizer_center, scheduler,
            loss_func, args.local_rank, dataset, teacher_trainloader=teacher_trainloader, teacher_dataset=teacher_dataset, 
            val_loader=val_loader, eval=args.eval, **kwargs)