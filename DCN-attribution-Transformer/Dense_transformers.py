"""This module defines the dense transformer components used in the study."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from functools import partial


def exists(val):
    return val is not None


class FeedForward(nn.Module):

    def __init__(self, dim, dim_out=None, mult=4, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


class SelfAttention(nn.Module):

    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        self.attention_weights = None                   

    def forward(self, x, return_attention=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        self.attention_weights = attn.detach()              

        attn = self.dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)

        if return_attention:
            return x, self.attention_weights
        return x


class TransformerBlock(nn.Module):

    def __init__(self, dim, num_heads=8, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SelfAttention(dim, num_heads=num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, return_attention=False):
        if return_attention:
            attn_out, attn_weights = self.attn(self.norm1(x), return_attention=True)
            x = x + attn_out
            x = x + self.mlp(self.norm2(x))
            return x, attn_weights
        else:
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
            return x


class Encoder(nn.Module):

    def __init__(self, dim, depth, heads, dim_head=64, mlp_dim=None, dropout=0.):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList([
            TransformerBlock(dim, num_heads=heads, mlp_ratio=4, dropout=dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, return_attention=False):
        attention_maps = []

        for block in self.layers:
            if return_attention:
                x, attn_weights = block(x, return_attention=True)
                attention_maps.append(attn_weights)
            else:
                x = block(x)

        x = self.norm(x)

        if return_attention:
            return x, attention_maps
        return x


class DTransformer(nn.Module):

    def __init__(self, *, image_size, patch_size, attn_layers,
                 num_classes, dropout=0.):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2      

                                             
        self.dim = attn_layers.dim if hasattr(attn_layers, 'dim') else 512

                                  
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, self.dim))

                             
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.dim))
        self.dropout = nn.Dropout(dropout)

        self.attn_layers = attn_layers
        self.norm = nn.LayerNorm(self.dim)

                 
        hidden_dim = max(128, self.dim // 2)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, img, return_attention=False):
        b, c, h, w = img.shape
        num_patches = h * w

                                                     
        patches = img.permute(0, 2, 3, 1).reshape(b, num_patches, -1)

                             
        if num_patches != self.num_patches:
            pos_embedding = F.interpolate(
                self.pos_embedding.permute(0, 2, 1),
                size=num_patches,
                mode='linear'
            ).permute(0, 2, 1)
        else:
            pos_embedding = self.pos_embedding

        patches = patches + pos_embedding

                           
        cls_token = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        patches = torch.cat((cls_token, patches), dim=1)                   
        patches = self.dropout(patches)

                        
        if return_attention:
            patches, attention_maps = self.attn_layers(patches, return_attention=True)
        else:
            patches = self.attn_layers(patches)

        patches = self.norm(patches)

                            
        cls_output = patches[:, 0]            
        logits = self.mlp_head(cls_output)

        if return_attention:
            return logits, attention_maps
        return logits
