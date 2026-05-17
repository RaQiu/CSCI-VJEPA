import math
from typing import Callable, Optional, Tuple, Union
from operator import mul
from functools import reduce

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from timm.layers import PatchEmbed, Mlp, GluMlp, SwiGLU, LayerNorm, DropPath, PatchDropout, RotaryEmbeddingCat, \
    apply_rot_embed_cat, apply_keep_indices_nlc, trunc_normal_, resample_patch_embed, resample_abs_pos_embed, \
    to_2tuple, use_fused_attn

from timm.models.helpers  import build_model_with_cfg
from timm.models.registry import generate_default_cfgs, register_model

from torchinfo import summary
from einops import rearrange, repeat

from model.utils import pooling

from model.eva_cloth_embed import EvaBlock as EvaBlock_img
from model.eva_cloth_embed import EvaAttention as EvaAttention_img
import random

__all__ = ['Eva', 'Eva_TA', 'EZ_Eva_Extra_tokens']


class Adapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)
        
    def forward(self, x):
        # x is (BT, HW+1, D)
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x




########## MODEL IGNORES CLOTHING LABELS ..... (not an input)
class Eva(nn.Module):
    """ Eva Vision Transformer w/ Abs & Rotary Pos Embed

    This class implements the EVA and EVA02 models that were based on the BEiT ViT variant
      * EVA - abs pos embed, global avg pool
      * EVA02 - abs + rope pos embed, global avg pool, SwiGLU, scale Norm in MLP (ala normformer)
    """

    def __init__(
            self,
            img_size: Union[int, Tuple[int, int]] = 224,
            patch_size: Union[int, Tuple[int, int]] = 16,
            in_chans: int = 3,
            num_classes: int = 1000,
            global_pool: str = 'avg',
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            qkv_bias: bool = True,
            qkv_fused: bool = True,
            mlp_ratio: float = 4.,
            swiglu_mlp: bool = False,
            scale_mlp: bool = False,
            scale_attn_inner: bool = False,
            drop_rate: float = 0.,
            pos_drop_rate: float = 0.,
            patch_drop_rate: float = 0.,
            proj_drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            norm_layer: Callable = LayerNorm,
            init_values: Optional[float] = None,
            class_token: bool = True,
            use_abs_pos_emb: bool = True,
            use_rot_pos_emb: bool = False,
            use_post_norm: bool = False,
            ref_feat_shape: Optional[Union[Tuple[int, int], int]] = None,
            head_init_scale: float = 0.001,
            cloth: int = 0,
            cloth_xishu: int = 3,
            tim_dim = 4,
            joint=None, 
            num_head_img_tokens=1, 
            sep_attn_for_img=None, ):
        """

        Args:
            img_size:
            patch_size:
            in_chans:
            num_classes:
            global_pool:
            embed_dim:
            depth:
            num_heads:
            qkv_bias:
            qkv_fused:
            mlp_ratio:
            swiglu_mlp:
            scale_mlp:
            scale_attn_inner:
            drop_rate:
            pos_drop_rate:
            proj_drop_rate:
            attn_drop_rate:
            drop_path_rate:
            norm_layer:
            init_values:
            class_token:
            use_abs_pos_emb:
            use_rot_pos_emb:
            use_post_norm:
            ref_feat_shape:
            head_init_scale:
        """
        super().__init__()
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_prefix_tokens = 1 if class_token else 0
        self.grad_checkpointing = False

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        self.cloth_xishu = cloth_xishu
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + self.num_prefix_tokens, embed_dim)) if use_abs_pos_emb else None
        # self.cloth_embed = nn.Parameter(torch.zeros(cloth, 1, embed_dim))
        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        if patch_drop_rate > 0:
            self.patch_drop = PatchDropout(
                patch_drop_rate,
                num_prefix_tokens=self.num_prefix_tokens,
                return_indices=True,
            )
        else:
            self.patch_drop = None

        if use_rot_pos_emb:
            feat_shape = self.patch_embed.grid_size
            if joint: 
                feat_shape = ref_feat_shape
            ref_feat_shape = to_2tuple(ref_feat_shape) if ref_feat_shape is not None else None
            self.rope = RotaryEmbeddingCat(
                embed_dim // num_heads,
                in_pixels=False,
                feat_shape=feat_shape,
                ref_feat_shape=ref_feat_shape,
            )
        else:
            self.rope = None

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        block_fn = EvaBlockPostNorm if use_post_norm else EvaBlock_time
        if joint: block_fn = EvaBlock_joint
        self.blocks = nn.ModuleList([
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qkv_fused=qkv_fused,
                mlp_ratio=mlp_ratio,
                swiglu_mlp=swiglu_mlp,
                scale_mlp=scale_mlp,
                scale_attn_inner=scale_attn_inner,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                init_values=init_values,
                tim_dim=tim_dim, 
                num_head_img_tokens=num_head_img_tokens,
                sep_attn_for_img= sep_attn_for_img, 
            )
            for i in range(depth)])

        use_fc_norm = self.global_pool == 'avg'
        self.norm = nn.Identity() if use_fc_norm else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)
        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)
        # trunc_normal_(self.cloth_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)

        self.fix_init_weight()
        if isinstance(self.head, nn.Linear):
            trunc_normal_(self.head.weight, std=.02)
            self.head.weight.data.mul_(head_init_scale)
            self.head.bias.data.mul_(head_init_scale)

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    @torch.jit.ignore
    def no_weight_decay(self):
        nwd = {'pos_embed', 'cls_token'}
        return nwd

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(
            stem=r'^cls_token|pos_embed|patch_embed',  # stem and embed
            blocks=[(r'^blocks\.(\d+)', None), (r'^norm', (99999,))],
        )
        return matcher

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            self.global_pool = global_pool
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x,cloth_id):
        x = self.patch_embed(x)

        if self.cls_token is not None:
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)

        # apply abs position embedding
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        # obtain shared rotary position embedding and apply patch dropout
        rot_pos_embed = self.rope.get_embed() if self.rope is not None else None
        if self.patch_drop is not None:
            x, keep_indices = self.patch_drop(x)
            if rot_pos_embed is not None and keep_indices is not None:
                rot_pos_embed = apply_keep_indices_nlc(x, rot_pos_embed, keep_indices)

        for blk in self.blocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(blk, x, rope=rot_pos_embed)
            else:
                x = blk(x, rope=rot_pos_embed)

        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool:
            x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x,cloth_id):
        x = self.forward_features(x,cloth_id)
        feat = self.forward_head(x, pre_logits=True)
        if not self.training:
            return feat
        else:
            cls_score = self.head(feat)
            return cls_score, feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))




class EvaAttention_joint(EvaAttention_img):
    def forward(self, x, rope: Optional[torch.Tensor] = None, attn_mask: Optional[torch.Tensor] = None,):

        B,N,C = x.shape

        if self.qkv is not None:
            qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
            qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
            qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
        else:
            q = self.q_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)  # B, num_heads, N, C
            k = self.k_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)
            v = self.v_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)

        if rope is not None:
            q = torch.cat([q[:, :, :5, :], apply_rot_embed_cat(q[:, :, 5:, :], rope)], 2).type_as(v)
            k = torch.cat([k[:, :, :5, :], apply_rot_embed_cat(k[:, :, 5:, :], rope)], 2).type_as(v)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p,
            )
        else:
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))
            attn = attn.softmax(dim=-1)
            if attn_mask is not None:
                attn_mask = attn_mask.to(torch.bool)
                attn = attn.masked_fill(~attn_mask[:, None, None, :], float("-inf"))
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.norm(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class EvaAttention_sep_joint(EvaAttention_img):
    
    def forward_sep( self, x, rope: Optional[torch.Tensor] = None, attn_mask: Optional[torch.Tensor] = None, ):
        B, N, C = x.shape

        if self.qkv is not None:
            qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
            qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
            qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
        else:
            q = self.q_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)  # B, num_heads, N, C
            k = self.k_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)
            v = self.v_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)

        if rope is not None:
            q = torch.cat([q[:, :, :self.num_head_tokens, :], apply_rot_embed_cat(q[:, :, self.num_head_tokens:, :], rope)], 2).type_as(v)
            k = torch.cat([k[:, :, :self.num_head_tokens, :], apply_rot_embed_cat(k[:, :, self.num_head_tokens:, :], rope)], 2).type_as(v)

        assert (attn_mask is None) or attn_mask[0] is None

        x_spatial = 0 
        heads = []
        if self.fused_attn:

            k_spatial_tokens = k[:,:, self.num_head_tokens:,:]
            v_spatial_tokens = v[:,:, self.num_head_tokens:,:]
            q_spatial_tokens = q[:,:, self.num_head_tokens:,:]
            

            for nth_head_token in range(self.num_head_tokens):
                k_nth_head= k[:,:,nth_head_token,:] 
                v_nth_head= v[:,:,nth_head_token,:]
                q_nth_head= q[:,:,nth_head_token,:]

                x_nth_head = F.scaled_dot_product_attention(
                    torch.cat([q_nth_head.unsqueeze(2), q_spatial_tokens], 2),
                    torch.cat([k_nth_head.unsqueeze(2), k_spatial_tokens], 2),
                    torch.cat([v_nth_head.unsqueeze(2), v_spatial_tokens], 2),
                    attn_mask=attn_mask,
                    dropout_p=self.attn_drop.p,
                )
                # x_extra_token, x_extra_spatial  = x_extra[:,:,0,:], x_extra[:,:,1:,:]
                x_nth_head_token, x_nth_head_spatial  = x_nth_head[:,:,0,:], x_nth_head[:,:,1:,:]
                x_spatial += x_nth_head_spatial
                heads.append(x_nth_head_token.unsqueeze(2))

        else:
            q = q * self.scale

            k_spatial_tokens = k[:,:, self.num_head_tokens:,:]
            v_spatial_tokens = v[:,:, self.num_head_tokens:,:]
            q_spatial_tokens = q[:,:, self.num_head_tokens:,:]
             
            for nth_head_token in range(self.num_head_tokens):
                k_nth_head= k[:,:,nth_head_token,:] 
                v_nth_head= v[:,:,nth_head_token,:]
                q_nth_head= q[:,:,nth_head_token,:]

                attn_nth_head = ( torch.cat([q_nth_head.unsqueeze(2), q_spatial_tokens], 2) @  torch.cat([k_nth_head.unsqueeze(2), k_spatial_tokens], 2).transpose(-2, -1))
                attn_nth_head = attn_nth_head.softmax(dim=-1)
                
                if attn_mask is not None:
                    attn_mask = attn_mask.to(torch.bool)
                    attn_nth_head = attn_nth_head.masked_fill(~attn_mask[:, None, None, :], float("-inf"))

                attn_nth_head = self.attn_drop(attn_nth_head)
                x_nth_head = attn_nth_head @ torch.cat([v_nth_head.unsqueeze(2), v_spatial_tokens], 2)

                x_nth_head_token, x_nth_head_spatial  = x_nth_head[:,:,0,:], x_nth_head[:,:,1:,:]
                # x_nth_head_token = x_nth_head_token * 0 + nth_head_token
                x_spatial += x_nth_head_spatial
                heads.append(x_nth_head_token.unsqueeze(2))
   
        x_spatial = x_spatial / self.num_head_tokens
        x = torch.cat(heads + [x_spatial], 2)
    
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.norm(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def forward(self, x, rope: Optional[torch.Tensor] = None, attn_mask: Optional[torch.Tensor] = None, use_sep_attn=False):
        if use_sep_attn:
            return self.forward_sep(x, rope, attn_mask)
        else:
            return super().forward(x, rope, attn_mask)

class EvaAttention_masked_joint(EvaAttention_sep_joint):
    
    def __init__(self, dim=None, **kwargs):
        super().__init__(dim=dim, **kwargs)
        
        HEIGHT=256
        NUM_HEAD_TOKENS = 2
        self.attn_mask = torch.zeros(HEIGHT + NUM_HEAD_TOKENS, HEIGHT + NUM_HEAD_TOKENS, requires_grad=False)
        for nth_head_token in range(NUM_HEAD_TOKENS):
            self.attn_mask[nth_head_token][:nth_head_token] += float("-inf")
            self.attn_mask[nth_head_token][nth_head_token+1:NUM_HEAD_TOKENS]  += float("-inf")
        
    def forward_sep( self, x, rope: Optional[torch.Tensor] = None, attn_mask: Optional[torch.Tensor] = None, ):
        B, N, C = x.shape

        if self.qkv is not None:
            qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
            qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
            qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
        else:
            q = self.q_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)  # B, num_heads, N, C
            k = self.k_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)
            v = self.v_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)

        if rope is not None:
            q = torch.cat([q[:, :, :self.num_head_tokens, :], apply_rot_embed_cat(q[:, :, self.num_head_tokens:, :], rope)], 2).type_as(v)
            k = torch.cat([k[:, :, :self.num_head_tokens, :], apply_rot_embed_cat(k[:, :, self.num_head_tokens:, :], rope)], 2).type_as(v)

        x_spatial = 0 
        heads = []
        assert attn_mask is None, "Mask is implemented on the spot"
        
        q = q * self.scale

        attn = (q @ k.transpose(-2, -1))
        
        attn_mask = self.attn_mask.unsqueeze(0).unsqueeze(0)
        attn_mask = attn_mask.to(attn.device)
        # attn_mask[:,:,0], attn_mask[:,:,1], attn_mask[:,:,2], attn_mask[:,:,3]
        attn += attn_mask
        attn = attn.softmax(dim=-1)
        
        # attn[random.choice(range(attn.shape[0])), random.choice(range(attn.shape[1])) ,0]
        # attn[random.choice(range(attn.shape[0])), random.choice(range(attn.shape[1])) ,1]
        # attn[random.choice(range(attn.shape[0])), random.choice(range(attn.shape[1])) ,2]
        
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.norm(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



class EvaBlock_time(EvaBlock_img):

    def __init__(self, dim: int, proj_drop: float = 0., tim_dim=4, use_adapter=False, num_head_img_tokens=1, sep_attn_for_img=None, 
            num_heads=0,  qkv_bias: bool = True,  qkv_fused: bool = True,  attn_drop: float = 0., 
            attn_head_dim: Optional[int] = None, norm_layer: Callable = LayerNorm,  scale_attn_inner: bool = False, num_head_tokens=1, **kwargs):
        super().__init__(dim=dim, proj_drop=proj_drop, num_heads=num_heads, qkv_bias=qkv_bias, qkv_fused=qkv_fused, 
            attn_drop=attn_drop, attn_head_dim=attn_head_dim, norm_layer=norm_layer, scale_attn_inner=scale_attn_inner, **kwargs)

        self.sep_attn_for_img = sep_attn_for_img
        self.num_head_img_tokens = num_head_img_tokens 

        ATTN_METHOD= EvaAttention_sep_joint
        if sep_attn_for_img:
            print(" ***  Sep Attention for Images and Joint for video .....  *** ")
        else:
            print(" ***  MASKED Attention for Images & Joint for video .....  *** ")
            ATTN_METHOD = EvaAttention_masked_joint
        
        del self.attn
        self.attn = ATTN_METHOD( dim, 
            num_heads=num_heads, 
            qkv_bias=qkv_bias, 
            qkv_fused=qkv_fused, 
            attn_drop=attn_drop, 
            proj_drop=proj_drop,  
            attn_head_dim=attn_head_dim,
            norm_layer=norm_layer if scale_attn_inner else None, 
            num_head_tokens=num_head_tokens, 
        )

        
        val = math.sqrt(6. / float(3 * reduce(mul, (16,16), 1) + dim))
        self.tim_dim = tim_dim
        self.temporal_prompt_embeddings = nn.Parameter(torch.randn(1, self.tim_dim, dim))
        nn.init.uniform_(self.temporal_prompt_embeddings.data, -val, val)
        self.temporal_dropout = nn.Dropout(proj_drop)

        self.use_adapter = use_adapter
        if use_adapter:
            self.Adapter = temporal_Adapter(dim)

    def video_forward(self, x, rope: Optional[torch.Tensor] = None, attn_mask: Optional[torch.Tensor] = None):
        T_prompt= self.temporal_dropout( self.temporal_prompt_embeddings ) .to(x.dtype)
        BT, N, C = x.shape
        B = BT // self.tim_dim

        x = rearrange(x , "(B T) N C -> N B T C", B= B, T=self.tim_dim)
        T_prompt = T_prompt + x.mean(0)
        
        if self.gamma_1 is None:
            T_prompt = self.drop_path1(self.attn(self.norm1(T_prompt), rope=None, attn_mask=None))
        else:
            T_prompt = self.drop_path1(self.gamma_1 * self.attn(self.norm1(T_prompt), rope=None, attn_mask=None))
        
        T_prompt = T_prompt.unsqueeze(0) # 1, B, T, C
        x = torch.cat([T_prompt, x, ], dim=0)
        x = rearrange(x , "N B T C -> (B T) N C")

        if self.gamma_1 is None:
            x = x + self.drop_path1(self.attn(self.norm1(x), rope=rope, attn_mask=attn_mask))
            x = x[:,1:]
            if self.use_adapter: x = self.Adapter(x)
            x = x + self.drop_path2(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), rope=rope, attn_mask=attn_mask))
            x = x[:,1:]
            if self.use_adapter: x = self.Adapter(x)
            x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
        return x

    def image_forward(self, x, rope: Optional[torch.Tensor] = None, attn_mask: Optional[torch.Tensor] = None, **kwargs):
        B, N, C = x.shape
        if self.gamma_1 is None:
            x = x + self.drop_path1(self.attn(self.norm1(x), rope=rope, attn_mask=attn_mask, **kwargs))
            x = x + self.drop_path2(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), rope=rope, attn_mask=attn_mask, **kwargs))
            x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
        return x

    def forward(self, x, rope: Optional[torch.Tensor] = None, attn_mask: Optional[torch.Tensor] = None, image_mode=None):
        if image_mode:
            self.attn.num_head_tokens = self.num_head_img_tokens
            # return self.image_forward(x=x, rope=rope, attn_mask=attn_mask, use_sep_attn=self.sep_attn_for_img)
            return self.image_forward(x=x, rope=rope, attn_mask=attn_mask, use_sep_attn=True)
        else:
            self.attn.num_head_tokens = 2
            return self.video_forward(x=x, rope=rope, attn_mask=attn_mask)
        
class EvaBlock_joint(EvaBlock_time):

    def __init__( self, dim: int, norm_layer: Callable = LayerNorm, scale_attn_inner: bool = False, 
        attn_head_dim: Optional[int] = None, **kwargs):
        super().__init__(dim=dim, norm_layer=norm_layer, scale_attn_inner=scale_attn_inner, attn_head_dim=attn_head_dim, **kwargs)
        self.attn = EvaAttention_joint(
            dim,
            num_heads=kwargs['num_heads'],
            qkv_bias=kwargs['qkv_bias'],
            qkv_fused=kwargs['qkv_fused'],
            attn_drop=kwargs['attn_drop'],
            proj_drop=kwargs['proj_drop'],
            attn_head_dim=attn_head_dim,
            norm_layer=norm_layer if scale_attn_inner else None,)
        
    def forward(self, x, rope: Optional[torch.Tensor] = None, attn_mask: Optional[torch.Tensor] = None):
        # x : B , 1 + (N T), C
        
        T_prompt= self.temporal_dropout( self.temporal_prompt_embeddings ) .to(x.dtype)
        
        cls_tokens = x[:, 0, :].unsqueeze(1)
        x = x[:,1:]
        B, NT, C = x.shape
        N = NT // self.tim_dim

        x = rearrange(x , "B (N T) C -> N B T C", B= B, T=self.tim_dim, N=N)
        T_prompt = T_prompt + x.mean(0) + cls_tokens.mean(1).unsqueeze(1)
        if self.gamma_1 is None:
            T_prompt = self.drop_path1(self.attn(self.norm1(T_prompt), rope=None, attn_mask=None))
        else:
            T_prompt = self.drop_path1(self.gamma_1 * self.attn(self.norm1(T_prompt), rope=None, attn_mask=None))
        
        # x : [N, B, T, C])
        x = rearrange(x , "N B T C -> B (N T) C")
        x = torch.cat([cls_tokens , x, ], dim=1)
        x = torch.cat([T_prompt, x], dim=1)
        
        if self.gamma_1 is None:
            x = x + self.drop_path1(self.attn(self.norm1(x), rope=rope, attn_mask=attn_mask))
            x = x[:,4:]
            x = x + self.drop_path2(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), rope=rope, attn_mask=attn_mask))
            x = x[:,4:]
            x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
        return x

class EZ_Eva(Eva):
    # https://github.com/Shahzadnit/EZ-CLIP/blob/main/clip/model.py
    def __init__(self, tim_dim=4, e2e_train = True, joint=False, ref_feat_shape=None, temporal_avg=None, spatial_avg=None, 
            norm_layer: Callable = LayerNorm, head_init_scale: float = 0.001, config=None, num_head_img_tokens=1, sep_attn_for_img=None, **kwargs):
        
        if joint:
            ref_feat_shape = ref_feat_shape[0] * int(tim_dim ** 0.5), ref_feat_shape[1] * int(tim_dim ** 0.5)
        super().__init__(tim_dim = tim_dim, joint=joint, ref_feat_shape=ref_feat_shape, norm_layer=norm_layer, head_init_scale=head_init_scale, num_head_img_tokens=num_head_img_tokens, 
            sep_attn_for_img=sep_attn_for_img, **kwargs)
        
        self.T = tim_dim
        L = len(self.blocks)
        self.temporal_tokens = nn.Parameter(torch.zeros(1, tim_dim, self.embed_dim))

        self.joint = joint
        if joint:
            scale = self.embed_dim ** -0.5
            print('===== using joint space-time ====')
            self.temporal_time_embedding = nn.Parameter(scale * torch.randn(tim_dim, self.embed_dim))
            # ref_feat_shape = (ref_feat_shape[0] * 2, ref_feat_shape[0] * 2)
            assert tim_dim == 4
            self.num_prefix_tokens = 0 

        self.e2e_train = e2e_train
        if not e2e_train :
            print("Trainable..... ")
            for name, p in self.named_parameters():
                if 'temporal' not in name :
                    p.requires_grad = False
                else:
                    print(name)
        
        if spatial_avg: assert False, "Spatial Pool Enabled, Img model failed on this...... "
        self.use_temporal_avg = temporal_avg
        if temporal_avg == 'max-avg':
            self.temporal_avg = pooling.MaxAvgPooling1D()
            self.fc_norm = norm_layer(self.embed_dim * 2) 
            self.head = nn.Linear(self.embed_dim * 2, self.num_classes)
            self._init_weights(self.head)
            self._init_weights(self.fc_norm)
            # self.fix_init_weight()
            if isinstance(self.head, nn.Linear):
                trunc_normal_(self.head.weight, std=.02)
                self.head.weight.data.mul_(head_init_scale)
                self.head.bias.data.mul_(head_init_scale)

        self.student_mode = None 
        self.teacher_mode = None 

        
        # self.rope.feat_shape (16, 16)
        # self.rope.ref_feat_shape (16, 16)
        # self.rope.max_res 224

    def forward_features(self, x,cloth_id):
        B,C,T,H,W = x.shape

        x = rearrange(x ," B C T H W -> (B T) C H W")
        x = self.patch_embed(x)

        if self.cls_token is not None:
            x = torch.cat((self.cls_token.expand(B * T, -1, -1), x), dim=1)
        
        # apply abs position embedding
        if self.pos_embed is not None:
            x = x + self.pos_embed
        

        N = x.shape[1]
        x = rearrange(x ,"(B T) N C -> (B N) T C", T=T, N=N)
        x = x + self.temporal_tokens.to(x.dtype)
        x = rearrange(x, '(B N) T C -> (B T) N C', N=N)

        if self.joint:
            cls_tokens = x[:B, 0, :].unsqueeze(1)
            x = x[:,1:]
            x = rearrange(x, '(B T) N C -> (B N) T C', B=B, T=self.T)
            x = x + self.temporal_time_embedding.to(x.dtype)
            x = rearrange(x, '(B N) T C -> B (N T) C', B=B, T=self.T)
            x = torch.cat((cls_tokens, x), dim=1)
            
        x = self.pos_drop(x)

        # obtain shared rotary position embedding and apply patch dropout
        rot_pos_embed = self.rope.get_embed() if self.rope is not None else None
        if self.patch_drop is not None:
            x, keep_indices = self.patch_drop(x)
            if rot_pos_embed is not None and keep_indices is not None:
                rot_pos_embed = apply_keep_indices_nlc(x, rot_pos_embed, keep_indices)

        for blk in self.blocks:
            x = blk(x, rope=rot_pos_embed)
            
        x = self.norm(x)
        return x, B,T

    def forward_head(self, x, B, T, pre_logits: bool = False):
        if self.global_pool:
            x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        
        x = rearrange(x , " (B T) C -> B T C", B=B, T=T)
        x_h = x.mean(1)    
        
        x_h = self.fc_norm(x_h)
        x_h = self.head_drop(x_h)
        return x, x_h

    def forward_head_temporal(self, x, B, T, pre_logits: bool = False):
        x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = rearrange(x , " (B T) C -> B C T", B=B, T=T)

        x_h = self.temporal_avg(x).squeeze(-1)
        x = rearrange(x , " B C T -> B T C")

        x_h = self.fc_norm(x_h)
        x_h = self.head_drop(x_h)
        return x, x_h

    def forward(self, x,cloth_id):
        x, B, T = self.forward_features(x,cloth_id)
        if self.joint:
            x = x[:,1:]
            N = x.shape[1] // T
            x = rearrange(x, 'B (N T) C -> (B T) N C', B=B, T=self.T, N=N)
        
        if self.use_temporal_avg == 'max-avg':
            feat, feat_h = self.forward_head_temporal(x, B, T, pre_logits=True)
        else:
            feat, feat_h = self.forward_head(x, B, T, pre_logits=True)
        if not self.training:
            return feat_h
        else:
            cls_score = self.head(feat_h)
            return cls_score, [feat_h, feat]

    def load_param(self, trained_path, load_head=True):
        logger = logging.getLogger('EVA-attribure')
        if self.e2e_train:logger.exception ("\n Why load pretrained weights when training e2e \n" )
        if  (self.use_temporal_avg or (not load_head)) :
            logger.exception ("\n Skipping HEAD \n" )
        param_dict = torch.load(trained_path, map_location='cpu')
        for i in param_dict:
            if (self.use_temporal_avg or (not load_head)) and (
                ("module.head.weight" == i ) or ("module.head.bias" == i ) or ("head.weight" == i ) or ("head.bias" == i ) or ("head_image.weight" == i ) or ("head_image.bias" == i )
               ):
                print("===", i )
                continue 
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))
    


def checkpoint_filter_fn_temporal(
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

def _create_eva(variant, pretrained=False, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Eva models.')

    model = build_model_with_cfg(
        EZ_Eva, variant, pretrained,
        pretrained_filter_fn=checkpoint_filter_fn_temporal,
        **kwargs)
    return model


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': OPENAI_CLIP_MEAN, 'std': OPENAI_CLIP_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        'license': 'mit', **kwargs
    }


default_cfgs = generate_default_cfgs({
    'eva02_large_patch14_224.mim_in22k': _cfg(
        #hf_hub_id='Yuxin-CV/EVA-02', hf_hub_filename='eva02/pt/eva02_L_pt_in21k_p14.pt',
        hf_hub_id='timm/',
        num_classes=0,
    ),
    'eva02_large_patch14_224.mim_m38m': _cfg(
        #hf_hub_id='Yuxin-CV/EVA-02', hf_hub_filename='eva02/pt/eva02_L_pt_m38m_p14.pt',
        hf_hub_id='timm/',
        num_classes=0,
    ),
    'eva02_large_patch14_clip_224.merged2b': _cfg(
        # hf_hub_id='QuanSun/EVA-CLIP', hf_hub_filename='EVA02_CLIP_L_psz14_s4B.pt',
        hf_hub_id='timm/eva02_large_patch14_clip_224.merged2b_s4b_b131k',  # float16 weights
        hf_hub_filename='open_clip_pytorch_model.bin',
        num_classes=768,
    ),
    'eva02_base_patch16_clip_224.merged2b': _cfg(
        # hf_hub_id='QuanSun/EVA-CLIP', hf_hub_filename='EVA02_CLIP_L_psz14_s4B.pt',
        hf_hub_id='timm/eva02_base_patch16_clip_224.merged2b_s8b_b131k',  # float16 weights
        hf_hub_filename='open_clip_pytorch_model.bin',
        num_classes=512,
    ),
})

model_args = dict(
        img_size=224,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4 * 2 / 3,
        qkv_fused=False,
        swiglu_mlp=True,
        scale_mlp=True,
        scale_attn_inner=True,
        use_rot_pos_emb=True,
        ref_feat_shape=(16, 16),  # 224/14
        cloth=300,
    )

@register_model
def ez_eva02_vid(pretrained=False, **kwargs) -> Eva:
    """ A EVA-CLIP specific variant that adds additional attn scale layernorm to eva02_large """
    model_args['global_pool'] = kwargs.pop('global_pool', 'token')
    model = _create_eva('eva02_large_patch14_clip_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model




if __name__ == "__main__":
    x = torch.randn([2, 3, 4, 224, 224])
    cloth_id = torch.tensor([2, 3])


    model = ez_eva02_vid(pretrained=True)
    output = model(x, cloth_id)
    print(output[0].shape)
    # print(summary(model, input_size=((8, 3, 224, 224),(3,7))))

# cd ~/MADE_ReID/model/
# python ez_eval_cloth_vid.py