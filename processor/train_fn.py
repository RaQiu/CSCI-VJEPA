import logging
import torch
from torch.nn import functional as F

from torch.cuda import amp
import numpy as np
from torchvision.utils import save_image 

from utils.meter import AverageMeter
from einops import rearrange, repeat

from tools.utils import normalize, rearrange_mlr, expand_input, reverse_arrange


def setup_grad_cam (model , use_wrapper=None, n_head_tokens=2, chosen_index=-1, DUMP_FOLDER=None):
    def reshape_transform(tensor, height=16, width=16, n_head_tokens=n_head_tokens):
        print(tensor.shape)
        result = tensor[:, n_head_tokens :  , :].reshape(tensor.size(0), height, width, tensor.size(2))
        result = result.transpose(2, 3).transpose(1, 2)
        return result
    
    class SimilarityToConceptTarget:
        def __init__(self, features):
            self.features = features
        
        def __call__(self, model_output):
            cos = torch.nn.CosineSimilarity(dim=0)
            return cos(model_output, self.features)

    """ Model wrapper to return a tensor"""
    class logit_wrapper(torch.nn.Module):
        def __init__(self, model, chosen_index=1):
            super(logit_wrapper, self).__init__()
            self.model = model
            self.chosen_index = chosen_index
            self.model.training = True 

        def forward(self, x):
            # return feature
            return self.model(x)[chosen_index]
            # return class logits 
            # return self.model(x)[0]

    model.train()
    print("****", model.training)
    model = logit_wrapper(model, chosen_index)
    print("****", model.model.training)

    import os 
    os.system(f"rm {DUMP_FOLDER}/*")
    from pytorch_grad_cam import GradCAM, DeepFeatureFactorization, GradCAMPlusPlus

    target_layer=([model.model.blocks[-1].norm1])
    # target_layer=([model.model.fc_norm])
    # target_layer=([model.model.norm])

    # target_layer=([model.head])
    # target_layer=([model.norm2])
    # model.grad_cam = "extra_token"

    # target_layer=([model.norm2 ])
    # target_layer=([model.head, model.mlp]) 
    
    # cam = DeepFeatureFactorization(model=model, target_layer=model.model.blocks[-1], computation_on_concepts=model.model.head)
    cam = GradCAM(model=model,  target_layers=target_layer, reshape_transform=reshape_transform)
    # cam = GradCAMPlusPlus(model=model,  target_layers=target_layer, reshape_transform=reshape_transform)
    # cam.batch_size = train_loader.batch_size   

    for param in model.model.parameters():
        param.requires_grad = True 
    return model, cam, SimilarityToConceptTarget

def default_img_loader_w_aux(cfg, data, ):
    text = None
    samples, targets, camids, _, clothes, meta, aux_info = data
    meta = None
    samples = samples.cuda(non_blocking=True)
    targets = targets.cuda(non_blocking=True)
    clothes = clothes.cuda(non_blocking=True)
    aux_info = aux_info.cuda(non_blocking=True)
    
    return samples, targets, clothes, meta, camids, text, aux_info

# from train import default_img_loader
def default_img_loader_wo_aux(cfg, data, ):
    text = None
    samples, targets, camids, _, clothes, meta = data
    meta = None
    samples = samples.cuda(non_blocking=True)
    targets = targets.cuda(non_blocking=True)
    clothes = clothes.cuda(non_blocking=True)
    
    return samples, targets, clothes, meta, camids, text


def cuda_eucledian_dist(x, y):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    dist = torch.sum(x ** 2, 1).unsqueeze(1) + torch.sum(y ** 2, 1).unsqueeze(
        1).transpose(0, 1) - 2 * torch.matmul(x, y.transpose(0, 1))
    dist = F.relu(dist)
    return dist

def handle_replica_data(imgs, pids, clothes, camids, ):
    B = imgs.shape[0]
    N_replicas = imgs.shape[1]
    
    pids = expand_input(pids, N_replicas)
    clothes = expand_input(clothes, N_replicas)
    camids = expand_input(camids, N_replicas)

    imgs = rearrange_mlr(imgs)
    # save_image(normalize(imgs), "t1.png"    
    return imgs, pids, clothes, camids, B, N_replicas 


def train_step_pair(cfg, model, train_loader, optimizer, optimizer_center, loss_fn, scaler, loss_meter, acc_meter, scheduler, epoch, DEFAULT_LOADER=None, train_writer=None, pair_loss=None, training_mode="image"):
    log_period = cfg.SOLVER.LOG_PERIOD
    logger = logging.getLogger("EVA-attribure.train")

    model.train()
    for idx, data in enumerate(train_loader):

        samples, targets, clothes, _, camids, _ = DEFAULT_LOADER(cfg, data, )
        samples, targets, clothes, camids, B, N_replicas = handle_replica_data(samples, targets, clothes, camids, )
        # save_image(normalize(samples), "t1.png" )   

        optimizer.zero_grad()
        optimizer_center.zero_grad()
        with amp.autocast(enabled=True):
            score, feat = model(samples, clothes)
        loss = loss_fn(score, feat, targets, camids, training_mode=training_mode)

        # samples = reverse_arrange(samples, B , N_replicas)
        feat = reverse_arrange(feat, B , N_replicas)
        pair_error = pair_loss(x = feat[:,0] , y = feat[:,1:] )
        loss += pair_error
        # save_image(normalize(samples[:,0]), "t1.png" )  , save_image(normalize(samples[:,1]), "t2.png" )    

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
    



def update_model(cfg, loss, epoch, scaler, optimizer, center_criterion=None, optimizer_center=None):
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
    
def update_metric(score, targets, loss, loss_meter, acc_meter, size=0):
    if isinstance(score, list):
        acc = (score[0].max(1)[1] == targets).float().mean()
    else:
        acc = (score.max(1)[1] == targets).float().mean()

    loss_meter.update(loss.item(), size)
    acc_meter.update(acc, 1)

    
def train_step_labels_dump(cfg, model, train_loader, optimizer, optimizer_center, loss_fn, scaler, loss_meter, acc_meter, scheduler, epoch, DEFAULT_LOADER=default_img_loader_w_aux, train_writer=None , training_mode="image", **kwargs):
    log_period = cfg.SOLVER.LOG_PERIOD
    logger = logging.getLogger("EVA-attribure.train")
    DUMP_FOLDER= "Samples2"
    N = len(train_loader)
    grad_cam = cfg.GRAD_CAM
    if grad_cam:
        from PIL import Image
        import cv2
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        model, cam, SimilarityToConceptTarget = setup_grad_cam (model , use_wrapper=True, n_head_tokens=1, chosen_index=1, DUMP_FOLDER=DUMP_FOLDER)
        model.model.training = True 

    model.train()
    feats = []
    pids= []
    clothes_labels = []
    for idx, data in enumerate(train_loader):
        # 2228 , len(data)
        samples, targets, clothes, meta, camids, text, index = DEFAULT_LOADER(cfg, data, )
        
        optimizer.zero_grad()
        if grad_cam:
            with amp.autocast(enabled=True):
                rgb_images = normalize(samples).cpu()
                feats = model (samples).detach()

                for i in range(len(rgb_images)-1):
                    pid, cloth_id, feats_x, x, rgb_img  = targets[i], clothes[i], feats[i], samples[i], rgb_images[i]
                    eligible = pid == targets[i+1:]
                    diff_cloth_eligible = eligible & (cloth_id != clothes[i+1:] )
                    if eligible.sum() < 2:continue
                    if diff_cloth_eligible.sum() == 0: continue 

                    N_candidates = eligible.sum()
                    feats_y = feats[i+1:][eligible]
                    rgb_img_y = rgb_images[i+1:][eligible.cpu()]
                    semi_label_x = [SimilarityToConceptTarget(feats_x)]
                    Image.fromarray( (rgb_img.numpy().transpose(1,2,0) * 255).astype(np.uint8) ).save(f'{DUMP_FOLDER}/{idx}_{i}_x.jpg')

                    for k in range(N_candidates):
                        semi_label_y = [SimilarityToConceptTarget(feats_y[k])]
                        # semi_label_y[0](feats_x)
                    
                        # labels = [ClassifierOutputTarget(targets[i].item())]
                        # grayscale_cam = cam(input_tensor=, targets=labels)
                        # concepts, batch_explanations, concept_scores = cam(samples[i].unsqueeze(0), n_components=5)
                        # grayscale_cam = cam(input_tensor=samples[i].unsqueeze(0), targets=labels, eigen_smooth=False, aug_smooth=False)

                        grayscale_cam = cam(input_tensor=x.unsqueeze(0), targets=semi_label_y)
                        # cam.outputs.shape

                        # import pdb; pdb.set_trace()
                        # grayscale_cam = cam(input_tensor=samples, targets=None, eigen_smooth=False, aug_smooth=False)
                        cam_image = show_cam_on_image(rgb_img.numpy().transpose(1,2,0), grayscale_cam[0], image_weight=0.5)

                        cv2.imwrite(f'{DUMP_FOLDER}/{idx}_{i}_{k}.jpg', cam_image)
                        Image.fromarray( (rgb_img_y[k].numpy().transpose(1,2,0) * 255).astype(np.uint8) ).save(f'{DUMP_FOLDER}/{idx}_{i}_{k}_y.jpg')
        else:
            with amp.autocast(enabled=True):
                score, feat = model(samples)
            feats.append(feat.cpu().detach())
            pids.append(targets.cpu().detach())
            clothes_labels.append(clothes.cpu().detach())
        print(f"{idx / N * 100:0.2f}, {idx}, {N}" , end="\r")
        torch.cuda.synchronize()
    if not grad_cam: 
        feats = torch.cat([e for e in feats])    
        pids = torch.cat([e for e in pids])    
        clothes_labels = torch.cat([e for e in clothes_labels])    
        from tools.utils import save_pickle
        feature_dump = dict(feats=feats.cpu(), pids=pids, clothes_labels=clothes_labels)
        save_pickle(feature_dump, cfg.TAG)    

    quit()
    
    

#### COLORS     
def train_w_color_labels_dump(cfg, model, train_loader, optimizer, optimizer_center, loss_fn, scaler, loss_meter, acc_meter, scheduler, epoch, DEFAULT_LOADER=default_img_loader_w_aux, train_writer=None, 
    mse=None, distentangle=None, ce=None, center_criterion=None, cosine=None, training_mode="image", color_loss_fn=None, **kwargs):

    log_period = cfg.SOLVER.LOG_PERIOD
    logger = logging.getLogger("EVA-attribure.train")
    DUMP_FOLDER= "Samples2"
    N = len(train_loader)
    grad_cam = cfg.GRAD_CAM
    if grad_cam:
        from PIL import Image
        import cv2
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        model, cam, SimilarityToConceptTarget = setup_grad_cam (model , use_wrapper=True, n_head_tokens=2, chosen_index=1, DUMP_FOLDER=DUMP_FOLDER)
    
    model.model.training = True 
    model.train()
    feats = []
    pids= []
    clothes_labels = []
    color_labels = []
    color_feat = []
    for idx, data in enumerate(train_loader):
        # len(train_loader), len(data)
        
        samples, targets, clothes, meta, camids, text, color_label= DEFAULT_LOADER(cfg, data, )
        optimizer.zero_grad()
        if grad_cam:
            # if idx % 10 !=0 :
            #     continue 
            with amp.autocast(enabled=True):
                rgb_images = normalize(samples).cpu()
                feats = model (samples).detach()
                for i in range(len(rgb_images)-1):
                    pid, cloth_id, feats_x, x, rgb_img  = targets[i], clothes[i], feats[i], samples[i], rgb_images[i]
                    eligible = pid == targets[i+1:]
                    diff_cloth_eligible = eligible & (cloth_id != clothes[i+1:] )
                    if eligible.sum() < 2:continue
                    if diff_cloth_eligible.sum() == 0: continue 
                    
                    # print(eligible, rgb_images[i+1:].shape)
                    # model.training, model.model.training

                    N_candidates = eligible.sum()
                    feats_y = feats[i+1:][eligible]
                    rgb_img_y = rgb_images[i+1:][eligible.cpu()]
                    semi_label_x = [SimilarityToConceptTarget(feats_x)]
                    Image.fromarray( (rgb_img.numpy().transpose(1,2,0) * 255).astype(np.uint8) ).save(f'{DUMP_FOLDER}/{idx}_{i}_x.jpg')

                    for k in range(N_candidates):
                        semi_label_y = [SimilarityToConceptTarget(feats_y[k])]
                        # semi_label_y[0](feats_x)
                    
                        # labels = [ClassifierOutputTarget(targets[i].item())]
                        # grayscale_cam = cam(input_tensor=, targets=labels)
                        # concepts, batch_explanations, concept_scores = cam(samples[i].unsqueeze(0), n_components=5)
                        # grayscale_cam = cam(input_tensor=samples[i].unsqueeze(0), targets=labels, eigen_smooth=False, aug_smooth=False)

                        grayscale_cam = cam(input_tensor=x.unsqueeze(0), targets=semi_label_y)
                        # cam.outputs.shape

                        # import pdb; pdb.set_trace()
                        # grayscale_cam = cam(input_tensor=samples, targets=None, eigen_smooth=False, aug_smooth=False)
                        cam_image = show_cam_on_image(rgb_img.numpy().transpose(1,2,0), grayscale_cam[0], image_weight=0.5)

                        cv2.imwrite(f'{DUMP_FOLDER}/{idx}_{i}_{k}.jpg', cam_image)
                        Image.fromarray( (rgb_img_y[k].numpy().transpose(1,2,0) * 255).astype(np.uint8) ).save(f'{DUMP_FOLDER}/{idx}_{i}_{k}_y.jpg')

        else:
            with amp.autocast(enabled=True):
                score, feat, color_output, color_feats, dist_loss = model(samples, clothes)
                # save_image(samples, "test.png")
                feats.append(feat.cpu().detach())
                pids.append(targets.cpu().detach())
                color_feat.append(color_feats.cpu())
                color_labels.append(color_label.cpu().detach())
                clothes_labels.append(clothes.cpu().detach())

        print(f"{idx / N * 100:0.2f}, {idx}, {N}" , end="\r")
        torch.cuda.synchronize()
    if not grad_cam:
        feats = torch.cat([e for e in feats])    
        pids = torch.cat([e for e in pids])    
        color_labels = torch.cat([e for e in color_labels]).squeeze()    
        clothes_labels = torch.cat([e for e in clothes_labels])    
        color_feat = torch.cat([e for e in color_feat])    

        from tools.utils import save_pickle
        feature_dump = dict(feats=feats.cpu(), pids=pids, color_labels=color_labels, clothes_labels=clothes_labels, color_feat=color_feat)
        save_pickle(feature_dump, cfg.TAG)
    quit()

        

def train_w_color_labels(cfg, model, train_loader, optimizer, optimizer_center, loss_fn, scaler, loss_meter, acc_meter, scheduler, epoch, DEFAULT_LOADER=default_img_loader_w_aux, train_writer=None, 
    mse=None, distentangle=None, ce=None, center_criterion=None, cosine=None, training_mode="image", color_loss_fn=None, TRAIN_DUMP=None, **kwargs):
    log_period = cfg.SOLVER.LOG_PERIOD
    logger = logging.getLogger("EVA-attribure.train")
    model.train()
    N = len(train_loader)

    for idx, data in enumerate(train_loader):

        samples, targets, clothes, meta, camids, text, color_label= DEFAULT_LOADER(cfg, data, )
        
        optimizer.zero_grad()
        optimizer_center.zero_grad()
        with amp.autocast(enabled=True):
            score, feat, color_output, color_feats, dist_loss = model(samples, clothes)
            # save_image(samples, "test.png")
            color_loss = color_loss_fn(color_output.float(), color_label.squeeze()).mean()

        loss = loss_fn(score, feat, targets, camids, training_mode=training_mode)
        loss += color_loss + dist_loss

        update_model(cfg, loss, epoch, scaler, optimizer, center_criterion=center_criterion, optimizer_center=optimizer_center)
        update_metric(score, targets, loss, loss_meter, acc_meter, size=samples.shape[0])

        torch.cuda.synchronize()
        if (idx + 1) % log_period == 0:
            logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Color: {:.3f}  Dist : {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                        .format(epoch, (idx + 1), len(train_loader),
                                loss_meter.avg, color_loss.mean().item(), dist_loss.mean().item(),  acc_meter.avg, scheduler._get_lr(epoch)[0]))
    return idx 

def train_w_color_labels_feed(cfg, model, train_loader, optimizer, optimizer_center, loss_fn, scaler, loss_meter, acc_meter, scheduler, epoch, DEFAULT_LOADER=default_img_loader_w_aux, train_writer=None, 
    mse=None, distentangle=None, ce=None, center_criterion=None, cosine=None, training_mode="image", color_loss_fn=None, **kwargs):
    log_period = cfg.SOLVER.LOG_PERIOD
    logger = logging.getLogger("EVA-attribure.train")
    model.train()
    N = len(train_loader)

    for idx, data in enumerate(train_loader):

        samples, targets, clothes, meta, camids, text, color_label= DEFAULT_LOADER(cfg, data, )
        
        optimizer.zero_grad()
        optimizer_center.zero_grad()
        with amp.autocast(enabled=True):
            score, feat, color_output, color_feats, dist_loss = model(samples, color_label, clothes)
            
        loss = loss_fn(score, feat, targets, camids, training_mode=training_mode)
        loss += dist_loss

        update_model(cfg, loss, epoch, scaler, optimizer, center_criterion=center_criterion, optimizer_center=optimizer_center)
        update_metric(score, targets, loss, loss_meter, acc_meter, size=samples.shape[0])

        torch.cuda.synchronize()
        if (idx + 1) % log_period == 0:
            logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Dist : {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                        .format(epoch, (idx + 1), len(train_loader),
                                loss_meter.avg, dist_loss.mean().item(),  acc_meter.avg, scheduler._get_lr(epoch)[0]))
    return idx 

def train_w_cl_dist(cfg, model, train_loader, optimizer, optimizer_center, loss_fn, scaler, loss_meter, acc_meter, scheduler, epoch, DEFAULT_LOADER=default_img_loader_wo_aux, train_writer=None, 
    mse=None, distentangle=None, ce=None, center_criterion=None, cosine=None, training_mode="image", color_loss_fn=None, TRAIN_DUMP=None, **kwargs):
    log_period = cfg.SOLVER.LOG_PERIOD
    logger = logging.getLogger("EVA-attribure.train")
    model.train()
    N = len(train_loader)

    for idx, data in enumerate(train_loader):

        # data[0], data[1], data[2], data[3], data[4], data[5] 
        samples, targets, clothes, meta, camids, text= DEFAULT_LOADER(cfg, data, )
        
        optimizer.zero_grad()
        optimizer_center.zero_grad()
        with amp.autocast(enabled=True):
            score, feat, color_output, color_feats, dist_loss = model(samples, clothes)
            # save_image(samples, "test.png")
            clothes_loss = color_loss_fn(color_output, clothes).mean()

        loss = loss_fn(score, feat, targets, camids, training_mode=training_mode)
        loss += clothes_loss + dist_loss

        update_model(cfg, loss, epoch, scaler, optimizer, center_criterion=center_criterion, optimizer_center=optimizer_center)
        update_metric(score, targets, loss, loss_meter, acc_meter, size=samples.shape[0])

        torch.cuda.synchronize()
        if (idx + 1) % log_period == 0:
            logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Clothes: {:.3f}  Dist : {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                        .format(epoch, (idx + 1), len(train_loader),
                                loss_meter.avg, clothes_loss.mean().item(), dist_loss.mean().item(),  acc_meter.avg, scheduler._get_lr(epoch)[0]))
    return idx 

def train_w_color_direct(cfg, model, train_loader, optimizer, optimizer_center, loss_fn, scaler, loss_meter, acc_meter, scheduler, epoch, DEFAULT_LOADER=default_img_loader_w_aux, train_writer=None, 
    mse=None, distentangle=None, ce=None, center_criterion=None, cosine=None, training_mode="image", color_loss_fn=None, **kwargs):
    log_period = cfg.SOLVER.LOG_PERIOD
    logger = logging.getLogger("EVA-attribure.train")
    model.train()
    N = len(train_loader)

    for idx, data in enumerate(train_loader):
        samples, targets, clothes, meta, camids, text, color_label= DEFAULT_LOADER(cfg, data, )
        
        optimizer.zero_grad()
        optimizer_center.zero_grad()
        with amp.autocast(enabled=True):
            score, feat, dist_loss = model(samples, color_label, clothes)
        
        loss = loss_fn(score, feat, targets, camids, training_mode=training_mode)
        loss += dist_loss

        update_model(cfg, loss, epoch, scaler, optimizer, center_criterion=center_criterion, optimizer_center=optimizer_center)
        update_metric(score, targets, loss, loss_meter, acc_meter, size=samples.shape[0])

        torch.cuda.synchronize()
        if (idx + 1) % log_period == 0:
            logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Dist : {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                        .format(epoch, (idx + 1), len(train_loader),
                                loss_meter.avg, dist_loss.mean().item(),  acc_meter.avg, scheduler._get_lr(epoch)[0]))
    return idx 


