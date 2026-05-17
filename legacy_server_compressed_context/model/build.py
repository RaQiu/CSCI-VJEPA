from .eva_meta import eva02_large_patch14_clip_224_meta,eva02_base_patch16_clip_224_meta
from .eva_meta_clothid import eva02_large_patch14_clip_224_meta_cloth
from .eva_cloth_embed import *
from .eva_cloth_vid import eva02_brute_3D

from .ez_eval_cloth_vid import ez_eva02_vid
from .ez_eva_custom import * 

__factory = {
    'eva02_meta_b_meta': eva02_base_patch16_clip_224_meta,
    'eva02_l_meta': eva02_large_patch14_clip_224_meta,
    'eva02_meta_cloth_l':eva02_large_patch14_clip_224_meta_cloth,
    'eva02_l_cloth':eva02_large_patch14_clip_224_cloth,
    
    'eva02_l_noclip':eva02_large_patch14_224_cloth, 
    'eva02_base_cloth':eva02_base_cloth, 
    

    'eva02_vid_TA': eva02_vid_TA,
    'eva02_brute_3D': eva02_brute_3D, 
    'ez_eva02_vid': ez_eva02_vid, 
    ## IMAGE 
    'eva02_img_extra_token': eva02_img_extra_token, 
    'eva02_img_extra_token_attribute' : eva02_img_extra_token_attribute, 
    'eva02_img_extra_token_feed': eva02_img_extra_token_feed, 
    'eva02_img_extra_token_CL' : eva02_img_extra_token_CL, 
    'eva02_img_extra_token_base': eva02_img_extra_token_base, 

    'eva02_img_no_token_color_mse': eva02_img_no_token_color_mse, 
    'eva02_img_no_token_color_mse_project_reid': eva02_img_no_token_color_mse_project_reid, 
    
    ## VIDEO
    'ez_eva02_vid_hybrid': ez_eva02_vid_hybrid,
    'ez_eva02_vid_hybrid_extra': ez_eva02_vid_hybrid_extra, 
    
    'EZ_Eva_T1_vid': EZ_Eva_T1_vid, 
    'Eva_img_ST': Eva_img_ST, 

}

def build_model(config,num_classes,cloth_num):
    model_type = config.MODEL.TYPE
    # config.MODEL.NAME :: 'eva02_l_cloth' , model_type == False
    pretrained = True 
    if not config.MODEL.PRETRAIN:
        print(" \n\n LOADING W/O PRETRAINING .... \n\n")
        pretrained = False 

    if 'vid' in config.MODEL.NAME:
        kwargs = dict(tim_dim=config.MODEL.TIM_DIM, e2e_train=config.TRAIN.E2E, joint=config.MODEL.Joint, spatial_avg=config.MODEL.SPATIAL_AVG, temporal_avg=config.MODEL.TEMPORAL_AVG)
        model = __factory[config.MODEL.NAME](config=config, pretrained=pretrained, num_classes=num_classes, cloth=cloth_num, cloth_xishu=config.MODEL.CLOTH_XISHU, **kwargs)
    elif model_type == 'eva02_meta_cloth':
        model = __factory[config.MODEL.NAME](config=config, pretrained=pretrained, num_classes=num_classes, meta_dims=config.MODEL.META_DIMS,cloth=cloth_num,cloth_xishu=config.MODEL.CLOTH_XISHU)
    else:
        model = __factory[config.MODEL.NAME](config=config, pretrained=pretrained, num_classes=num_classes,cloth=cloth_num,cloth_xishu=config.MODEL.CLOTH_XISHU, spatial_avg=config.MODEL.SPATIAL_AVG)
    return model
