# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .triplet_loss import TripletLoss
from .center_loss import CenterLoss
from loss.custom_loss import *
from .center_loss import CenterLoss
from functools import *

def loss_vid_func(score, feat, target, target_cam, cfg=None, triplet=None, training_mode="video"):
    feat, feat_t = feat # feat --> feat_h
    if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
        if isinstance(score, list):
            ID_LOSS = [F.cross_entropy(scor, target) for scor in score[1:]]
            ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
            ID_LOSS = 0.5 * ID_LOSS + 0.5 * F.cross_entropy(score[0], target)
        else:
            ID_LOSS = F.cross_entropy(score, target)
        if isinstance(feat, list):
                TRI_LOSS = [triplet(feats, target)[0] for feats in feat[1:]]
                TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
                TRI_LOSS = 0.5 * TRI_LOSS + 0.5 * triplet(feat[0], target)[0]
        else:
                TRI_LOSS = triplet(feat, target)[0]
        return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
                cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS

def loss_vid_func_ml(score, feat, target, target_cam, cfg=None, triplet=None, motion_loss=None, training_mode="video"):
    feat, feat_t = feat # feat --> feat_h
    ML_LOSS = motion_loss(feat_t)
    if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
        if isinstance(score, list):
            ID_LOSS = [F.cross_entropy(scor, target) for scor in score[1:]]
            ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
            ID_LOSS = 0.5 * ID_LOSS + 0.5 * F.cross_entropy(score[0], target)
        else:
            ID_LOSS = F.cross_entropy(score, target)
        if isinstance(feat, list):
                TRI_LOSS = [triplet(feats, target)[0] for feats in feat[1:]]
                TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
                TRI_LOSS = 0.5 * TRI_LOSS + 0.5 * triplet(feat[0], target)[0]
        else:
                TRI_LOSS = triplet(feat, target)[0]
        
        return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS +\
                ML_LOSS

def loss_vid_image_func_ml(score, feat, target, target_cam, cfg=None, triplet=None, motion_loss=None, training_mode="video"):
    assert cfg.MODEL.METRIC_LOSS_TYPE == 'triplet', "cfg.MODEL.METRIC_LOSS_TYPE != Triplet, WTF???? "

    ID_LOSS = F.cross_entropy(score, target)

    if training_mode == "image":
        TRI_LOSS = triplet(feat, target)[0]
        return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
    else:
        feat, feat_t = feat # feat --> feat_h
        ML_LOSS = motion_loss(feat_t)
        TRI_LOSS = triplet(feat, target)[0]
        
        return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS +\
                ML_LOSS


def make_loss(cfg, num_classes):    # modified by gu
    sampler = cfg.DATA.SAMPLER
    feat_dim = 1024
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss
    if 'triplet' in cfg.MODEL.METRIC_LOSS_TYPE:
        if cfg.MODEL.NO_MARGIN:
            triplet = TripletLoss()
            print("using soft triplet loss for training")
        else:
            triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
            print("using triplet loss with margin:{}".format(cfg.SOLVER.MARGIN))
    else:
        print('expected METRIC_LOSS_TYPE should be triplet'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        print("label smooth on, numclasses:", num_classes)

    if sampler == 'softmax':
        def loss_func(score, feat, target):
            return F.cross_entropy(score, target)

    elif cfg.DATA.SAMPLER == 'softmax_triplet':
        if cfg.TRAIN.HYBRID:
            assert cfg.TRAIN.TRAIN_VIDEO, "Motion loss with images? and no Video??? "
            ML_LOSS = Motion_loss(debug=cfg.TRAIN.DEBUG)
            loss_func = partial(loss_vid_image_func_ml, cfg = cfg, triplet=triplet, motion_loss=ML_LOSS)


        elif cfg.MODEL.MOTION_LOSS:
            assert cfg.TRAIN.TRAIN_VIDEO, "Motion loss with images? and no Video??? "
            ML_LOSS = Motion_loss(debug=cfg.TRAIN.DEBUG)
            loss_func = partial(loss_vid_func_ml, cfg = cfg, triplet=triplet, motion_loss=ML_LOSS)
        elif cfg.TRAIN.TRAIN_VIDEO:
            loss_func = partial(loss_vid_func, cfg = cfg, triplet=triplet)
        else:
            def loss_func(score, feat, target, target_cam, training_mode=None ):
                if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
                    if cfg.MODEL.IF_LABELSMOOTH == 'on':
                        if isinstance(score, list):
                            ID_LOSS = [xent(scor, target) for scor in score[1:]]
                            ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                            ID_LOSS = 0.5 * ID_LOSS + 0.5 * xent(score[0], target)
                        else:
                            ID_LOSS = xent(score, target)

                        if isinstance(feat, list):
                                TRI_LOSS = [triplet(feats, target)[0] for feats in feat[1:]]
                                TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
                                TRI_LOSS = 0.5 * TRI_LOSS + 0.5 * triplet(feat[0], target)[0]
                        else:
                                TRI_LOSS = triplet(feat, target)[0]

                        return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
                                cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
                    else:
                        if isinstance(score, list):
                            ID_LOSS = [F.cross_entropy(scor, target) for scor in score[1:]]
                            ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                            ID_LOSS = 0.5 * ID_LOSS + 0.5 * F.cross_entropy(score[0], target)
                        else:
                            ID_LOSS = F.cross_entropy(score, target)

                        if isinstance(feat, list):
                                TRI_LOSS = [triplet(feats, target)[0] for feats in feat[1:]]
                                TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
                                TRI_LOSS = 0.5 * TRI_LOSS + 0.5 * triplet(feat[0], target)[0]
                        else:
                                TRI_LOSS = triplet(feat, target)[0]

                        return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
                                cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
                else:
                    print('expected METRIC_LOSS_TYPE should be triplet'
                        'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    else:
        print('expected sampler should be softmax, triplet, softmax_triplet or softmax_triplet_center'
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func, center_criterion



