# Cross attention

"""
Mostly copy-paste from DINO and timm library:
https://github.com/facebookresearch/dino
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""

import math
import torch
import torch.nn as nn

from functools import partial
import sys
sys.path.append("/mnt/workspace/workgroup/shaofeng.zhang/pos_bot/")
from utils import trunc_normal_
from timm.models.registry import register_model
from models.pos_embed import get_2d_sincos_pos_embed, get_2d_local_sincos_pos_embed


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def global_forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

    def global_local_forward(self, x, y):
        B, N, C = x.shape
        L = y.shape[1]
        x = torch.cat([x, y], dim=1)
        qkv = self.qkv(x).reshape(B, N+L, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Cross attention
        attn = (q[:, :, N:] @ k[:, :, :].transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        y = (attn @ v[:, :, :]).transpose(1, 2).reshape(B, L, C)
        y = self.proj(y)
        y = self.proj_drop(y)

        # Self attention
        attn = (q[:, :, :N] @ k[:, :, :N].transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v[:, :, :N]).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, y, attn

    def forward(self, x, y=None):
        if y is None:
            return self.global_forward(x)
        else:
            return self.global_local_forward(x, y)

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, init_values=0):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None
    
    def global_forward(self, global_images, return_attention=False):
        y_global, attn = self.attn(self.norm1(global_images))
        if return_attention:
            return attn
        if self.gamma_1 is None:
            global_images = global_images + self.drop_path(y_global)
            global_images = global_images + self.drop_path(self.mlp(self.norm2(global_images)))
        else:
            global_images = global_images + self.drop_path(self.gamma_1 * y_global)
            global_images = global_images + self.drop_path(self.gamma_2 * self.mlp(self.norm2(global_images)))
        return global_images

    def global_local_forward(self, global_images, local_images, return_attention=False):
        y_global, y_local, attn = self.attn(self.norm1(global_images), self.norm1(local_images))
        if return_attention:
            return attn
        if self.gamma_1 is None:
            global_images = global_images + self.drop_path(y_global)
            global_images = global_images + self.drop_path(self.mlp(self.norm2(global_images)))
            local_images = local_images + self.drop_path(y_local)
            local_images = local_images + self.drop_path(self.mlp(self.norm2(local_images)))
        else:
            global_images = global_images + self.drop_path(self.gamma_1 * y_global)
            global_images = global_images + self.drop_path(self.gamma_2 * self.mlp(self.norm2(global_images)))
            local_images = local_images + self.drop_path(self.gamma_1 * y_local)
            local_images = local_images + self.drop_path(self.gamma_2 * self.mlp(self.norm2(local_images)))
        return global_images, local_images


    def forward(self, global_images, local_images=None, return_attention=False):
        if local_images is None:
            return self.global_forward(global_images, return_attention)
        else:
            return self.global_local_forward(global_images, local_images)
       


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        return self.proj(x)

class PosVisionTransformer(nn.Module):
    """ Vision Transformer """

    def __init__(self, img_size=[224], patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6), return_all_tokens=False,
                 init_values=0, use_mean_pooling=False, masked_im_modeling=False):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.return_all_tokens = return_all_tokens

        self.patch_embed = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])

        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.norm2 = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

        # masked image modeling
        self.masked_im_modeling = masked_im_modeling
        self.masked_embed = nn.Parameter(torch.zeros(1, embed_dim))

        self.query_block = Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0., norm_layer=norm_layer,
                init_values=init_values)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(math.sqrt(self.pos_embed.shape[1]-1)), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    def interpolate_pos_encoding(self, x, h, w):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-1] and int(h0) == patch_pos_embed.shape[-2]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens(self, global_images, mask=None):
        B, C, H, W = global_images.shape
        # patch linear embedding
        global_images = self.patch_embed(global_images)
        # mask image modeling
        if mask is not None:
            global_images = self.mask_model(global_images, mask)
        global_images = global_images.flatten(2).transpose(1, 2)

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        global_images = torch.cat((cls_tokens, global_images), dim=1)
        
        # add positional encoding to each token
        global_pos_embed = self.interpolate_pos_encoding(global_images, H, W)
        global_images = global_images + global_pos_embed

        return self.pos_drop(global_images)

    def t_forward(self, global_images):
        global_images = self.prepare_tokens(global_images)
        for blk in self.blocks:
            global_images = blk(global_images)
        global_images = self.norm(global_images)
        if self.fc_norm is not None:
            global_images[:, 0] = self.fc_norm(global_images[:, 1:, :].mean(1))
        return global_images
    
    def s_forward(self, global_images, local_pos, mask):
        global_images = self.prepare_tokens(global_images, mask)
        local_pos = torch.cat(local_pos, dim=1)
        local_images = local_pos + self.masked_embed.to(global_images.dtype)
        for blk in self.blocks:
            global_images, local_images = blk(global_images, local_images)
        global_images = self.norm(global_images)
        local_images = self.norm(local_images)
        if self.fc_norm is not None:
            global_images[:, 0] = self.fc_norm(global_images[:, 1:, :].mean(1))
        return global_images, local_images

    def forward(self, global_images, local_pos=None, mask=None):
        """
        local_pos: [local_view1, local_view2, ..., local_view10] list
        """
        if local_pos is None:
            return self.t_forward(global_images)
        else:
            return self.s_forward(global_images, local_pos, mask)

    def get_num_layers(self):
        return len(self.blocks)

    def mask_model(self, x, mask):
        x.permute(0, 2, 3, 1)[mask, :] = self.masked_embed.to(x.dtype)
        return x
    
    def get_intermediate_layers(self, x, n=1):
        x = self.prepare_tokens(x)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output


def pos_tiny(patch_size=16, **kwargs):
    model = PosVisionTransformer(
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, **kwargs)
    return model


def pos_small(patch_size=16, **kwargs):
    model = PosVisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, **kwargs)
    return model


def pos_base(patch_size=16, **kwargs):
    model = PosVisionTransformer(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, **kwargs)
    return model


def pos_large(patch_size=16, **kwargs):
    model = PosVisionTransformer(
        patch_size=patch_size, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4,
        qkv_bias=True, **kwargs)
    return model

# if __name__ == '__main__':
#     model_t = pos_small()
#     model_s = pos_small()
#     g_views = torch.randn([2, 3, 224, 224])
#     l_views = torch.randn([2, 3, 96, 96])
#     local_pos = [torch.randn([2, 36, 384])]
#     mask = torch.randint(0, 2, size=(1, 14, 14)).bool().repeat(2, 1, 1)
#     s_g, s_l = model_s(g_views, local_pos, mask=mask)
#     t_g = model_t(g_views)
#     t_l = model_t(l_views)
#     print(t_g.size(), t_l.size())
#     print(s_g.size(), s_l.size())
