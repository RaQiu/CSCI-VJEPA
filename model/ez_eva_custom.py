import torch 
import torch.nn as nn

from  model.eva_cloth_embed import Eva as Eva_Image
from  model.ez_eval_cloth_vid import EZ_Eva, build_model_with_cfg, register_model, \
    GluMlp, LayerNorm, trunc_normal_, resample_abs_pos_embed, SwiGLU

from model.eva_cloth_embed import checkpoint_filter_fn as checkpoint_filter_fn_vanilla
from model.utils.spatial_transformer import *
from tools.utils import save_image, normalize

from loss.custom_loss import Cosine_Disentangle

from model.ez_eval_cloth_vid import EZ_Eva
from model.ez_eval_cloth_vid import checkpoint_filter_fn_temporal as checkpoint_filter_fn

import logging

model_args = dict(
        img_size=224, patch_size=14, embed_dim=1024
        , depth=24, num_heads=16, 
        mlp_ratio=4 * 2 / 3, qkv_fused=False, swiglu_mlp=True, scale_mlp=True, 
        scale_attn_inner=True, use_rot_pos_emb=True, ref_feat_shape=(16, 16),  # 224/14
        # cloth=300,
    )


class JepaFeatureFusion(nn.Module):
    def __init__(self, embed_dim=1024, jepa_dim=1664, hidden_dim=4096, context_tokens=4, num_heads=8, num_layers=1, dropout=0.1):
        super().__init__()
        # context_tokens is kept for old configs; dense_patch fusion keeps all JEPA tokens.
        self.adapter = nn.Sequential(
            nn.LayerNorm(jepa_dim),
            nn.Linear(jepa_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.fusion = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_norm = nn.LayerNorm(embed_dim)

    def forward(self, tracklet_feat, frame_feats, rgb_patch_tokens, jepa_tokens):
        if jepa_tokens is None:
            return tracklet_feat
        jepa_tokens = jepa_tokens.detach().to(device=tracklet_feat.device, dtype=tracklet_feat.dtype)
        jepa_tokens = self.adapter(jepa_tokens)
        tokens = [tracklet_feat.unsqueeze(1)]
        if frame_feats is not None:
            tokens.append(frame_feats.to(device=tracklet_feat.device, dtype=tracklet_feat.dtype))
        if rgb_patch_tokens is not None:
            tokens.append(rgb_patch_tokens.to(device=tracklet_feat.device, dtype=tracklet_feat.dtype))
        tokens.append(jepa_tokens)
        return self.out_norm(self.fusion(torch.cat(tokens, dim=1))[:, 0])


class EZ_Eva_T1(EZ_Eva):
    def __init__(self, config=None, embed_dim=None, **kwargs):
        super().__init__(config=config, embed_dim=embed_dim, **kwargs)
        self.adapter = GluMlp(
            in_features=embed_dim, hidden_features=embed_dim * 2,  out_features=embed_dim * 4, 
            norm_layer=LayerNorm, act_layer=nn.SiLU, gate_last=False, drop=0)
        self.adapter_fc_norm = LayerNorm(embed_dim)
        self._init_weights(self.adapter)
                
    def forward(self, x,cloth_id):
        x, B, T = self.forward_features(x,cloth_id)
        feat, feat_h = self.forward_head(x, B, T, pre_logits=True)
        if self.student_mode: 
            adapt_feat = self.adapter_fc_norm(feat_h)
            adapt_feat = self.adapter(adapt_feat)
            cls_score = self.head(feat_h)
            return cls_score, [feat_h, feat, adapt_feat]

        # self.adapter()
        if not self.training:
            return feat_h
        else:
            cls_score = self.head(feat_h)
            return cls_score, [feat_h, feat]

class Eva_ST(Eva_Image):
    def __init__(self, config=None, depth= 12, embed_dim=768, **kwargs):
        super().__init__(depth=depth, embed_dim=embed_dim, **kwargs)
        
        self.debug = None 
        if config.TRAIN.DEBUG:
            self.debug = True 
        self.st1 = Spatial_transformers3( intermediate=20 )

        # self.sts = nn.ModuleList([
        #     Spatial_transformers2(
        #         Height=16, 
        #         Width=16, 
        #         intermediate=64,
        #         input_channel=embed_dim,
        #         # output_spatial_dim=16 * 16 * 64, 
        #     )
        #     for i in range(depth)])
        self.sts = None 

    def forward_features(self, x,cloth_id):
        if self.debug:
            import pdb
            pdb.set_trace()
            save_image(normalize(x[::3]), "temp.png")
            z = self.st1(x)
            save_image(normalize(z[::3]), "temp2.png")
            theta = self.st1.fc(x.reshape(x.shape[0], -1)).view(x.shape[0], 2, 3)
            self.st1.fc[0].weight
            self.st1.fc[3].weight
            self.st1.fc[3].bias
            y = self.st1.fc[0](x.reshape(x.shape[0], -1))
            y = self.st1.fc[1](y)
            y = self.st1.fc[3](y)
            quit()
        else:
            x = self.st1(x)
        

        x = self.patch_embed(x)

        if self.cls_token is not None:
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)

        # apply abs position embedding
        if self.pos_embed is not None:
            # if self.training:
            # x = x + self.pos_embed + self.cloth_xishu * self.cloth_embed[cloth_id]
            x = x + self.pos_embed
        x = self.pos_drop(x)

        # obtain shared rotary position embedding and apply patch dropout
        rot_pos_embed = self.rope.get_embed() if self.rope is not None else None
        if self.patch_drop is not None:
            x, keep_indices = self.patch_drop(x)
            if rot_pos_embed is not None and keep_indices is not None:
                rot_pos_embed = apply_keep_indices_nlc(x, rot_pos_embed, keep_indices)

        if self.sts is None :
            for blk in self.blocks:
                x = blk(x, rope=rot_pos_embed)
        else:
            for blk, st_blk in zip(self.blocks, self.sts):
                x = blk(x, rope=rot_pos_embed)
                x = st_blk(x, BNC=True)
                
        x = self.norm(x)
        return x


###### IMAGE 
class Eva_Extra_Token(Eva_Image):
    def __init__(self, config=None, embed_dim=768, mlp_ratio=None, num_head_tokens=2, layer_disentangle=None, **kwargs):
        extra_token_separation = True 
        if config.MODEL.UNIFIED_DIST:
            extra_token_separation = False 
        super().__init__(config=config, embed_dim=embed_dim, num_head_tokens=num_head_tokens, mlp_ratio=mlp_ratio, extra_token_separation=extra_token_separation, masked_sep_attn=config.MODEL.MASKED_SEP_ATTN, **kwargs)

        self.extra_token1 = nn.Parameter(torch.zeros(1, 1, embed_dim)) 
        trunc_normal_(self.extra_token1, std=.02)
        
        extra_token_dim = config.MODEL.EXTRA_DIM
        self.norm2 = LayerNorm( embed_dim )

        self.mlp = SwiGLU(
                    in_features=embed_dim,
                    out_features=  extra_token_dim, 
                    hidden_features=int(extra_token_dim * mlp_ratio),
                    norm_layer=LayerNorm,
                    drop=0.0,
                )
        self.layer_disentangle = config.TRAIN.LAYER_DISESNTANGLE
        self.distentangle = Cosine_Disentangle()

        self.student_mode = False 
        # self.teacher_mode
        self.grad_cam = None
        self.dump_aux = None
        self.early_return = None
        if config.MODEL.RETURN_EARLY is not None:
            self.early_return = config.MODEL.RETURN_EARLY

    def forward_features(self, x,cloth_id):
        distentangle_loss = 0 
        x = self.patch_embed(x)

        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = torch.cat((self.extra_token1.expand(x.shape[0], -1, -1), x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        # obtain shared rotary position embedding and apply patch dropout
        rot_pos_embed = self.rope.get_embed() if self.rope is not None else None
        if self.patch_drop is not None:
            x, keep_indices = self.patch_drop(x)
            if rot_pos_embed is not None and keep_indices is not None:
                rot_pos_embed = apply_keep_indices_nlc(x, rot_pos_embed, keep_indices)

        if (self.early_return is not None) and self.early_return == 0 :
            return x, distentangle_loss

        for i,blk in enumerate(self.blocks):
            x = blk(x, rope=rot_pos_embed)
            if self.early_return and self.early_return == i:
                return x, distentangle_loss
            if self.layer_disentangle:
                extra_token = x[:,0]
                class_token = x[:,1]
                distentangle_loss += self.distentangle(extra_token, class_token)

        x = self.norm(x)
        return x, distentangle_loss

    def forward(self, x,cloth_id=None):
        x, distentangle_loss = self.forward_features(x,cloth_id)
        extra_token_feats = x[:,0]
        default = x[:,1:]
        feat = self.forward_head(default, pre_logits=True)
        if not self.training:
            if self.dump_aux:
                extra_token = self.mlp(self.norm2(extra_token_feats))
                return feat , extra_token_feats, extra_token
            return feat
        else:
            cls_score = self.head(feat)
            if self.student_mode:
                return cls_score, feat
            else:
                extra_token = self.mlp(self.norm2(extra_token_feats))
            distentangle_loss += self.distentangle(extra_token_feats, feat)
            return cls_score, feat, extra_token, extra_token_feats, distentangle_loss
            # score, feat, color_output, color_feats, dist_loss = model(samples, clothes)

    def load_param(self, trained_path, load_head=True):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))


class Eva_Extra_Token_CL(Eva_Extra_Token):
    def __init__(self, config=None, cloth=-1, embed_dim=768, **kwargs):
        super().__init__(config=config, cloth=cloth, embed_dim=embed_dim, **kwargs)
        del self.mlp
        self.mlp = nn.Sequential( nn.Dropout(0.0), nn.Linear(embed_dim, cloth)  )
    
class Eva_Extra_Token_att(Eva_Extra_Token):
    def __init__(self, config=None, **kwargs):
        super().__init__(config=config, **kwargs)
        self.mlp = nn.Sequential(
            self.mlp,
            nn.Sigmoid(),
        )


class Eva_Extra_Token_Feed(Eva_Extra_Token):
    def __init__(self, config=None, embed_dim=768, mlp_ratio=None, num_head_tokens=2, layer_disentangle=None, **kwargs):
        super().__init__(config=config, embed_dim=embed_dim, num_head_tokens=num_head_tokens, mlp_ratio=mlp_ratio, **kwargs)
        extra_token_dim = config.MODEL.EXTRA_DIM
        self.norm2 = LayerNorm( extra_token_dim )
        self.mlp = SwiGLU(in_features= extra_token_dim, out_features= embed_dim, 
                    hidden_features=int(extra_token_dim * mlp_ratio), norm_layer=LayerNorm, drop=0.0, )
        
    def forward_features(self, x, extra_data, cloth_id=None):
        distentangle_loss = 0 
        x = self.patch_embed(x)

        if self.training :
            extra_data = self.mlp ( self.norm2(extra_data) )
        else:
            extra_data = self.cls_token.expand(x.shape[0], -1, -1) * 0
        
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)        
        x = torch.cat((extra_data, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        # obtain shared rotary position embedding and apply patch dropout
        rot_pos_embed = self.rope.get_embed() if self.rope is not None else None
        if self.patch_drop is not None:
            x, keep_indices = self.patch_drop(x)
            if rot_pos_embed is not None and keep_indices is not None:
                rot_pos_embed = apply_keep_indices_nlc(x, rot_pos_embed, keep_indices)

        for blk in self.blocks:
            x = blk(x, rope=rot_pos_embed)
            if self.layer_disentangle:
                extra_token = x[:,0]
                class_token = x[:,1]
                distentangle_loss += self.distentangle(extra_token, class_token)

        x = self.norm(x)
        return x, distentangle_loss

    def forward(self, x, extra_data=None, cloth_id=None):
        x, distentangle_loss = self.forward_features(x, extra_data, cloth_id)
        extra_token_feats = x[:,0]
        default = x[:,1:]
        feat = self.forward_head(default, pre_logits=True)
        if not self.training:
            return feat
        else:
            cls_score = self.head(feat)
            if self.student_mode:
                return cls_score, feat
            else:
                extra_token = None 
            distentangle_loss += self.distentangle(extra_token_feats, feat)
            return cls_score, feat, extra_token, extra_token_feats, distentangle_loss
            # score, feat, color_output, color_feats, dist_loss = model(samples, clothes)


class Eva_No_Extra_Token(Eva_Image):
    def __init__(self, config=None, embed_dim=768, **kwargs):
        extra_token_separation = False 
        super().__init__(config=config, embed_dim=embed_dim, extra_token_separation=extra_token_separation, **kwargs)

        self.distentangle = Cosine_Disentangle()

        self.student_mode = False 
        self.grad_cam = None
        self.dump_aux = None
        
    def forward(self, x, extra_data=None, cloth_id=None):
        x = self.forward_features(x,cloth_id)
        feat = self.forward_head(x, pre_logits=True)
        if not self.training:
            return feat
        else:
            cls_score = self.head(feat)
            if self.student_mode:
                return cls_score, feat
            else:
                extra_token = extra_data
            distentangle_loss = self.distentangle(extra_token.squeeze(), feat)
            return cls_score, feat, distentangle_loss

class Eva_No_Extra_Token_ProjectReID(Eva_No_Extra_Token):
    def __init__(self, config=None, embed_dim=768, mlp_ratio=None, **kwargs):
        super().__init__(config=config, embed_dim=embed_dim, mlp_ratio=mlp_ratio, **kwargs)
        
        
        extra_token_dim = config.MODEL.EXTRA_DIM
        self.norm2 = LayerNorm( embed_dim )

        self.mlp = SwiGLU(
                    in_features=embed_dim,
                    out_features=  extra_token_dim, 
                    hidden_features=int(extra_token_dim * mlp_ratio),
                    norm_layer=LayerNorm,
                    drop=0.0,
                )
        self.distentangle = Cosine_Disentangle()


    def forward(self, x, extra_data=None, cloth_id=None):
        x = self.forward_features(x,cloth_id)
        
        feat = self.forward_head(x, pre_logits=True)
        if not self.training:
            return feat
        else:
            cls_score = self.head(feat)
            if self.student_mode:
                return cls_score, feat
            else:
                extra_token = extra_data
            
            color_feats = self.mlp(self.norm2(feat))
            distentangle_loss = self.distentangle(extra_token.squeeze(), color_feats)
            return cls_score, feat, distentangle_loss

    


###### VIDEO
class EZ_Eva_Hybrid(EZ_Eva):
    def __init__(self, config=None, embed_dim=None, class_token: bool = True, head_init_scale: float = 0.001, **kwargs):
        super().__init__( config=config, embed_dim=embed_dim, class_token=class_token, head_init_scale=head_init_scale, **kwargs)
        
        num_patches = self.patch_embed.num_patches
        self.cls_token_img = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        trunc_normal_(self.cls_token_img, std=.02)
        image_num_classes = config.TRAIN.TEACH1_NUMCLASSES or kwargs.get("num_classes")
        if image_num_classes:
            self.head_image = nn.Linear(embed_dim, image_num_classes) 
        else:
            self.head_image = nn.Identity() 

        if isinstance(self.head_image, nn.Linear):
            trunc_normal_(self.head_image.weight, std=.02)
            self.head_image.weight.data.mul_(head_init_scale)
            self.head_image.bias.data.mul_(head_init_scale)

        self.jepa_fusion = None
        if config is not None and getattr(config.MODEL, "JEPA_ENABLE", False):
            if getattr(config.MODEL, "JEPA_FUSION_MODE", "dense_patch") != "dense_patch":
                raise ValueError("Only MODEL.JEPA_FUSION_MODE='dense_patch' is supported in the 1421-token JEPA path.")
            self.jepa_fusion = JepaFeatureFusion(
                embed_dim=embed_dim,
                jepa_dim=config.MODEL.JEPA_IN_DIM,
                hidden_dim=config.MODEL.JEPA_ADAPTER_HIDDEN,
                context_tokens=config.MODEL.JEPA_CONTEXT_TOKENS,
                num_heads=config.MODEL.JEPA_FUSION_HEADS,
                num_layers=config.MODEL.JEPA_FUSION_LAYERS,
            )

    def video_forward(self, x,cloth_id,jepa_tokens=None):
        x, B, T = self.forward_features(x,cloth_id)
        # Dense fusion uses the CSCI-V RGB patch tokens plus the V-JEPA latent tokens.
        if self.joint:
            rgb_patch_tokens = x[:, 1:]
        else:
            rgb_patch_tokens = x[:, self.num_prefix_tokens:]
            rgb_patch_tokens = rearrange(rgb_patch_tokens, "(B T) N C -> B (T N) C", B=B, T=T)
        feat, feat_h = self.forward_head(x, B, T, pre_logits=True)
        feat_h = self.jepa_fusion(feat_h, feat, rgb_patch_tokens, jepa_tokens) if self.jepa_fusion is not None else feat_h
        if not self.training:
            return feat_h
        else:
            cls_score = self.head(feat_h)
            return cls_score, [feat_h, feat]

    def image_forward_features(self, x,cloth_id):
        x = self.patch_embed(x)

        if self.cls_token_img is not None:
            x = torch.cat((self.cls_token_img.expand(x.shape[0], -1, -1), x), dim=1)

        # apply abs position embedding
        if self.pos_embed is not None:
            # if self.training:
            # x = x + self.pos_embed + self.cloth_xishu * self.cloth_embed[cloth_id]
            x = x + self.pos_embed
        x = self.pos_drop(x)

        # obtain shared rotary position embedding and apply patch dropout
        rot_pos_embed = self.rope.get_embed() if self.rope is not None else None
        if self.patch_drop is not None:
            x, keep_indices = self.patch_drop(x)
            if rot_pos_embed is not None and keep_indices is not None:
                rot_pos_embed = apply_keep_indices_nlc(x, rot_pos_embed, keep_indices)

        for blk in self.blocks:
            x = blk(x, rope=rot_pos_embed, image_mode=True)

        x = self.norm(x)
        return x

    def forward_image_head(self, x, pre_logits: bool = False):
        if self.global_pool:
            x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def image_forward(self, x,cloth_id,jepa_tokens=None):
        x = self.image_forward_features(x,cloth_id)
        feat = self.forward_image_head(x, pre_logits=True)
        rgb_patch_tokens = x[:, self.num_prefix_tokens:]
        feat = self.jepa_fusion(feat, None, rgb_patch_tokens, jepa_tokens) if self.jepa_fusion is not None else feat
        if not self.training:
            return feat
        else:
            cls_score = self.head_image(feat)
            return cls_score, feat
    
    def forward(self, x,cloth_id,jepa_tokens=None):
        if self.student_mode or x.dim() == 4:
            return self.image_forward(x,cloth_id,jepa_tokens=jepa_tokens)
        else:
            return self.video_forward(x,cloth_id,jepa_tokens=jepa_tokens)

class EZ_Eva_Extra_tokens_Pose(EZ_Eva_Hybrid):
    def __init__(self, config=None, embed_dim=None, mlp_ratio=None, num_head_tokens=2, sep_attn_for_img=True, extra_token_dim=None, **kwargs):
        super().__init__( config=config, embed_dim=embed_dim, mlp_ratio=mlp_ratio, num_head_img_tokens=num_head_tokens, sep_attn_for_img=sep_attn_for_img, **kwargs)
        
        self.extra_token1 = nn.Parameter(torch.zeros(1, 1, embed_dim)) 
        trunc_normal_(self.extra_token1, std=.02)
        
        if extra_token_dim is None:
            extra_token_dim = config.MODEL.EXTRA_DIM
        self.norm2 = LayerNorm( embed_dim )

        self.layer_disentangle = config.TRAIN.LAYER_DISESNTANGLE
        self.distentangle = Cosine_Disentangle()
        self.mlp = nn.Identity()

        self.extra_pos_in_temporal = nn.Parameter(torch.zeros(1, 1, embed_dim))
        trunc_normal_(self.extra_pos_in_temporal, std=.02)

    def image_forward_features(self, x,cloth_id):
        distentangle_loss = 0 
        x = self.patch_embed(x)
        
        x = torch.cat((self.cls_token_img.expand(x.shape[0], -1, -1), x), dim=1)
        x = torch.cat((self.extra_token1.expand(x.shape[0], -1, -1), x), dim=1)

        pos_embed = torch.cat([self.extra_pos_in_temporal, self.pos_embed], 1)

        x = x + pos_embed
        x = self.pos_drop(x)

        # obtain shared rotary position embedding and apply patch dropout
        rot_pos_embed = self.rope.get_embed() if self.rope is not None else None
        if self.patch_drop is not None:
            x, keep_indices = self.patch_drop(x)
            if rot_pos_embed is not None and keep_indices is not None:
                rot_pos_embed = apply_keep_indices_nlc(x, rot_pos_embed, keep_indices)

        for blk in self.blocks:
            x = blk(x, rope=rot_pos_embed, image_mode=True)
            if self.layer_disentangle:
                extra_token = x[:,0]
                class_token = x[:,1]
                distentangle_loss += self.distentangle(extra_token, class_token)

        x = self.norm(x)
        return x, distentangle_loss

    def forward_image_head(self, x, pre_logits: bool = False):
        if self.global_pool:
            x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def image_forward(self, x,cloth_id,jepa_tokens=None):
        x, distentangle_loss = self.image_forward_features(x,cloth_id)
        extra_token_feats = x[:,0]
        default = x[:,1:]
        feat = self.forward_image_head(default, pre_logits=True)
        rgb_patch_tokens = default[:, self.num_prefix_tokens:]
        feat = self.jepa_fusion(feat, None, rgb_patch_tokens, jepa_tokens) if self.jepa_fusion is not None else feat
        if not self.training:
            return feat
        else:
            cls_score = self.head_image(feat)
            extra_token = self.mlp(self.norm2(extra_token_feats))
            distentangle_loss += self.distentangle(extra_token_feats, feat)
            return cls_score, feat, extra_token, extra_token_feats, distentangle_loss
            
    def load_param(self, trained_path, load_head=True):
        super().load_param(trained_path=trained_path, load_head=load_head) 
        param_dict = torch.load(trained_path, map_location='cpu')
        if "extra_pos_in_temporal" in param_dict:
            if "pos_embed" in param_dict:
                self.state_dict()['extra_pos_in_temporal'].copy_( param_dict['extra_pos_in_temporal'][:,:1]  )    
            else:
                self.state_dict()['extra_pos_in_temporal'].copy_( param_dict['module.extra_pos_in_temporal'][:,:1]  )
        else:
            if "pos_embed" in param_dict:
                self.state_dict()['extra_pos_in_temporal'].copy_( param_dict['pos_embed'][:,:1]  )    
            else:
                self.state_dict()['extra_pos_in_temporal'].copy_( param_dict['module.pos_embed'][:,:1]  )
            
class EZ_Eva_Extra_tokens(EZ_Eva_Extra_tokens_Pose):
    def __init__(self, config=None, embed_dim=None, mlp_ratio=None, **kwargs):
        extra_token_separation = True 
        if config.MODEL.MASKED_SEP_ATTN:
            extra_token_separation = False 
        super().__init__( config=config, embed_dim=embed_dim, mlp_ratio=mlp_ratio, sep_attn_for_img=extra_token_separation,  **kwargs)
        extra_token_dim = config.MODEL.EXTRA_DIM

        self.mlp = SwiGLU(
            in_features=embed_dim,
            out_features=  extra_token_dim, 
            hidden_features=int(extra_token_dim * mlp_ratio),
            norm_layer=LayerNorm,
            drop=0.0,
        )
        self.student_mode = False 
    


def checkpoint_filter_fn_temporal_ext_token(
        state_dict,
        model,
        interpolation='bicubic',
        antialias=True,
    ):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    state_dict = state_dict.get('model_ema', state_dict)
    state_dict = state_dict.get('model', state_dict)
    state_dict = state_dict.get('module', state_dict)
    state_dict = state_dict.get('state_dict', state_dict)
    # prefix for loading OpenCLIP compatible weights
    if 'visual.trunk.pos_embed' in state_dict:
        prefix = 'visual.trunk.'
    elif 'visual.pos_embed' in state_dict:
        prefix = 'visual.'
    else:
        prefix = ''
    mim_weights = prefix + 'mask_token' in state_dict
    no_qkv = prefix + 'blocks.0.attn.q_proj.weight' in state_dict

    len_prefix = len(prefix)
    for k, v in state_dict.items():
        if prefix:
            if k.startswith(prefix):
                k = k[len_prefix:]
            else:
                continue
        if 'rope' in k:
            # fixed embedding no need to load buffer from checkpoint
            continue

        if 'patch_embed.proj.weight' in k:
            _, _, H, W = model.patch_embed.proj.weight.shape
            if v.shape[-1] != W or v.shape[-2] != H:
                v = resample_patch_embed(
                    v,
                    (H, W),
                    interpolation=interpolation,
                    antialias=antialias,
                    verbose=True,
                )
        elif k == 'pos_embed' and v.shape[1] != model.pos_embed.shape[1]:
            # To resize pos embedding when using model at different size from pretrained weights
            num_prefix_tokens = 0 if getattr(model, 'no_embed_class', False) else getattr(model, 'num_prefix_tokens', 1)
            v = resample_abs_pos_embed(
                v,
                new_size=model.patch_embed.grid_size,
                num_prefix_tokens=num_prefix_tokens,
                interpolation=interpolation,
                antialias=antialias,
                verbose=True,
            )

        if k == 'pos_embed':
            out_dict["extra_pos_in_temporal"] = v[:, : 1 ]            
            if getattr(model, 'extra_pos_in_temporal2', None) is not None :
                out_dict["extra_pos_in_temporal2"] = v[:, : 1 ]


        k = k.replace('mlp.ffn_ln', 'mlp.norm')
        k = k.replace('attn.inner_attn_ln', 'attn.norm')
        k = k.replace('mlp.w12', 'mlp.fc1')
        k = k.replace('mlp.w1', 'mlp.fc1_g')
        k = k.replace('mlp.w2', 'mlp.fc1_x')
        k = k.replace('mlp.w3', 'mlp.fc2')
        if no_qkv:
            k = k.replace('q_bias', 'q_proj.bias')
            k = k.replace('v_bias', 'v_proj.bias')

        if mim_weights and k in ('mask_token', 'lm_head.weight', 'lm_head.bias', 'norm.weight', 'norm.bias'):
            if k == 'norm.weight' or k == 'norm.bias':
                # try moving norm -> fc norm on fine-tune, probably a better starting point than new init
                k = k.replace('norm', 'fc_norm')
            else:
                # skip pretrain mask token & head weights
                continue

        out_dict[k] = v
    for key,val in model.state_dict().items():
        if 'temporal' in key:
            out_dict[key] = val
            

    return out_dict


def checkpoint_filter_fn_for_extra_token( state_dict, model, interpolation='bicubic', antialias=True, ):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    state_dict = state_dict.get('model_ema', state_dict)
    state_dict = state_dict.get('model', state_dict)
    state_dict = state_dict.get('module', state_dict)
    state_dict = state_dict.get('state_dict', state_dict)
    # prefix for loading OpenCLIP compatible weights
    if 'visual.trunk.pos_embed' in state_dict:
        prefix = 'visual.trunk.'
    elif 'visual.pos_embed' in state_dict:
        prefix = 'visual.'
    else:
        prefix = ''
    mim_weights = prefix + 'mask_token' in state_dict
    no_qkv = prefix + 'blocks.0.attn.q_proj.weight' in state_dict

    len_prefix = len(prefix)
    for k, v in state_dict.items():
        if prefix:
            if k.startswith(prefix):
                k = k[len_prefix:]
            else:
                continue

        if 'rope' in k:
            # fixed embedding no need to load buffer from checkpoint
            continue

        if 'patch_embed.proj.weight' in k:
            _, _, H, W = model.patch_embed.proj.weight.shape
            if v.shape[-1] != W or v.shape[-2] != H:
                v = resample_patch_embed(
                    v,
                    (H, W),
                    interpolation=interpolation,
                    antialias=antialias,
                    verbose=True,
                )
        elif k == 'pos_embed' and v.shape[1] != model.pos_embed.shape[1]:
            # To resize pos embedding when using model at different size from pretrained weights
            new_num_prefix_tokens = 0 if getattr(model, 'no_embed_class', False) else getattr(model, 'num_prefix_tokens', 1)
            origin_num_prefix_tokens = 0 if getattr(model, 'no_embed_class', False) else 1
            v = resample_abs_pos_embed( v, new_size=model.patch_embed.grid_size, num_prefix_tokens=origin_num_prefix_tokens, interpolation=interpolation, antialias=antialias, verbose=True, )

            prefix_tokens = v[:, : origin_num_prefix_tokens ]
            new_tokens = [prefix_tokens for e in range(new_num_prefix_tokens-origin_num_prefix_tokens)] + [v]
            v = torch.cat(new_tokens, dim=1)
            
        k = k.replace('mlp.ffn_ln', 'mlp.norm')
        k = k.replace('attn.inner_attn_ln', 'attn.norm')
        k = k.replace('mlp.w12', 'mlp.fc1')
        k = k.replace('mlp.w1', 'mlp.fc1_g')
        k = k.replace('mlp.w2', 'mlp.fc1_x')
        k = k.replace('mlp.w3', 'mlp.fc2')
        if no_qkv:
            k = k.replace('q_bias', 'q_proj.bias')
            k = k.replace('v_bias', 'v_proj.bias')

        if mim_weights and k in ('mask_token', 'lm_head.weight', 'lm_head.bias', 'norm.weight', 'norm.bias'):
            if k == 'norm.weight' or k == 'norm.bias':
                # try moving norm -> fc norm on fine-tune, probably a better starting point than new init
                k = k.replace('norm', 'fc_norm')
            else:
                # skip pretrain mask token & head weights
                continue

        out_dict[k] = v

    return out_dict



@register_model
def EZ_Eva_T1_vid(pretrained=False, **kwargs):
    model_args['global_pool'] = kwargs.pop('global_pool', 'token')
    variant= 'eva02_large_patch14_clip_224'
    kwargs.update(model_args)
    model = build_model_with_cfg(EZ_Eva_T1, variant, pretrained=pretrained, pretrained_filter_fn=checkpoint_filter_fn, **kwargs)
    return model

@register_model
def Eva_img_ST(pretrained=False, **kwargs):
    model_args['global_pool'] = kwargs.pop('global_pool', 'token')
    variant= 'eva02_large_patch14_clip_224'
    kwargs.update(model_args)
    model = build_model_with_cfg(Eva_ST, variant, pretrained=pretrained, pretrained_filter_fn=checkpoint_filter_fn, **kwargs)
    return model



###### IMAGE 
@register_model
def eva02_img_extra_token(pretrained=False, **kwargs):
    model_args['global_pool'] = kwargs.pop('global_pool', 'token')
    variant= 'eva02_large_patch14_clip_224'
    kwargs.update(model_args)
    model = build_model_with_cfg(Eva_Extra_Token, variant, pretrained=pretrained, pretrained_filter_fn=checkpoint_filter_fn_for_extra_token, **kwargs)
    return model

@register_model
def eva02_img_no_token_color_mse(pretrained=False, **kwargs):
    model_args['global_pool'] = kwargs.pop('global_pool', 'token')
    variant= 'eva02_large_patch14_clip_224'
    kwargs.update(model_args)
    model = build_model_with_cfg(Eva_No_Extra_Token, variant, pretrained=pretrained, pretrained_filter_fn=checkpoint_filter_fn_vanilla, **kwargs)
    return model

@register_model
def eva02_img_no_token_color_mse_project_reid(pretrained=False, **kwargs):
    model_args['global_pool'] = kwargs.pop('global_pool', 'token')
    variant= 'eva02_large_patch14_clip_224'
    kwargs.update(model_args)
    model = build_model_with_cfg(Eva_No_Extra_Token_ProjectReID, variant, pretrained=pretrained, pretrained_filter_fn=checkpoint_filter_fn_vanilla, **kwargs)
    return model



@register_model
def eva02_img_extra_token_base(pretrained=False, **kwargs):
    model_args['global_pool'] = kwargs.pop('global_pool', 'token')
    model_args['patch_size'] = 16
    model_args['embed_dim'] = 768
    model_args['depth'] = 12    
    model_args['num_heads'] = 12
    
    variant= 'eva02_base_patch16_clip_224'
    kwargs.update(model_args)
    model = build_model_with_cfg(Eva_Extra_Token, variant, pretrained=pretrained, pretrained_filter_fn=checkpoint_filter_fn_for_extra_token, **kwargs)
    return model




@register_model
def eva02_img_extra_token_attribute(pretrained=False, **kwargs):
    model_args['global_pool'] = kwargs.pop('global_pool', 'token')
    variant= 'eva02_large_patch14_clip_224'
    kwargs.update(model_args)
    model = build_model_with_cfg(Eva_Extra_Token_att, variant, pretrained=pretrained, pretrained_filter_fn=checkpoint_filter_fn_for_extra_token, **kwargs)
    return model

@register_model
def eva02_img_extra_dist_token_gender(pretrained=False, **kwargs):
    model_args['global_pool'] = kwargs.pop('global_pool', 'token')
    variant= 'eva02_large_patch14_clip_224'
    kwargs.update(model_args)
    model = build_model_with_cfg(Eva_Extra_Dist_Token_gender, variant, pretrained=pretrained, pretrained_filter_fn=checkpoint_filter_fn_for_extra_token, **kwargs)
    return model


@register_model
def eva02_img_extra_token_feed(pretrained=False, **kwargs):
    model_args['global_pool'] = kwargs.pop('global_pool', 'token')
    variant= 'eva02_large_patch14_clip_224'
    kwargs.update(model_args)
    model = build_model_with_cfg(Eva_Extra_Token_Feed, variant, pretrained=pretrained, pretrained_filter_fn=checkpoint_filter_fn_for_extra_token, **kwargs)
    return model


@register_model
def eva02_img_extra_token_CL(pretrained=False, **kwargs):
    model_args['global_pool'] = kwargs.pop('global_pool', 'token')
    variant= 'eva02_large_patch14_clip_224'
    kwargs.update(model_args)
    model = build_model_with_cfg(Eva_Extra_Token_CL, variant, pretrained=pretrained, pretrained_filter_fn=checkpoint_filter_fn_for_extra_token, **kwargs)
    return model






###### VIDEO 
# @register_model
# def ez_eva02_vid_extra(pretrained=False, **kwargs):
#     model_args['global_pool'] = kwargs.pop('global_pool', 'token')
#     model = _create_eva('eva02_large_patch14_clip_224', pretrained=pretrained, **dict(model_args, **kwargs))
#     return model

@register_model
def ez_eva02_vid_hybrid(pretrained=False, **kwargs):
    model_args['global_pool'] = kwargs.pop('global_pool', 'token')
    variant= 'eva02_large_patch14_clip_224'
    kwargs.update(model_args)
    model = build_model_with_cfg(EZ_Eva_Hybrid, variant, pretrained=pretrained, pretrained_filter_fn=checkpoint_filter_fn, **kwargs)
    return model

@register_model
def ez_eva02_vid_hybrid_pose(pretrained=False, **kwargs):
    model_args['global_pool'] = kwargs.pop('global_pool', 'token')
    variant= 'eva02_large_patch14_clip_224'
    kwargs.update(model_args)
    model = build_model_with_cfg(EZ_Eva_Extra_tokens_Pose, variant, pretrained=pretrained, pretrained_filter_fn=checkpoint_filter_fn_temporal_ext_token, **kwargs)
    return model

@register_model
def ez_eva02_vid_hybrid_extra(pretrained=False, **kwargs):
    model_args['global_pool'] = kwargs.pop('global_pool', 'token')
    variant= 'eva02_large_patch14_clip_224'
    kwargs.update(model_args)
    model = build_model_with_cfg(EZ_Eva_Extra_tokens, variant, pretrained=pretrained, pretrained_filter_fn=checkpoint_filter_fn_temporal_ext_token, **kwargs)
    return model

@register_model
def ez_eva02_vid_hybrid_pose_color(pretrained=False, **kwargs):
    model_args['global_pool'] = kwargs.pop('global_pool', 'token')
    variant= 'eva02_large_patch14_clip_224'
    kwargs.update(model_args)
    model = build_model_with_cfg(EZ_Eva_Extra_Pose_Color, variant, pretrained=pretrained, pretrained_filter_fn=checkpoint_filter_fn_temporal_ext_token, **kwargs)
    return model

@register_model
def ez_eva02_vid_hybrid_pose_color_contours(pretrained=False, **kwargs):
    model_args['global_pool'] = kwargs.pop('global_pool', 'token')
    variant= 'eva02_large_patch14_clip_224'
    kwargs.update(model_args)
    model = build_model_with_cfg(EZ_Eva_Extra_Pose_Color_Contours, variant, pretrained=pretrained, pretrained_filter_fn=checkpoint_filter_fn_temporal_ext_token, **kwargs)
    return model







    
    



    
    
