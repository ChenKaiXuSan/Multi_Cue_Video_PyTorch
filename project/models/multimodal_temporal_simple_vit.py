#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
File: /workspace/code/project/models/multimodal_temporal_simple_vit.py
Project: /workspace/code/project/models
Created Date: Wednesday June 18th 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Wednesday June 18th 2025 11:31:16 am
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
'''
import torch
from torch import nn
from einops import rearrange
from einops.layers.torch import Rearrange

# --- Helper Functions ---
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype=torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)

# --- Core Modules ---
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([
            nn.ModuleList([
                Attention(dim, heads, dim_head),
                FeedForward(dim, mlp_dim)
            ]) for _ in range(depth)
        ])

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, channels, dim):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, \
            "Image size must be divisible by patch size."
        
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        # p1, p2 = pair(patch_size)
        # patch_dim = channels * p1 * p2

        self.embed = nn.Sequential(
            Rearrange("b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)", p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.num_patches = num_patches

    def forward(self, x):  # [B, T, C, H, W]
        return self.embed(x)  # [B, T, N_patch, D]

class MultiModalTemporalViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim,
                 channels_dict, dim_head=64, num_frames=8):
        super().__init__()
        self.modalities = list(channels_dict.keys())
        self.embedders = nn.ModuleDict({
            name: PatchEmbedding(image_size, patch_size, channels, dim)
            for name, channels in channels_dict.items()
        })

        h, w = pair(image_size)
        ph, pw = pair(patch_size)
        self.num_patches = (h // ph) * (w // pw)
        self.total_tokens = self.num_patches * len(self.modalities)

        self.pos_embedding = nn.Parameter(
            posemb_sincos_2d(h // ph, w // pw, dim).repeat(len(self.modalities), 1)
        )  # [N_modal * N_patch, D]

        self.temporal_embedding = nn.Parameter(
            torch.randn(1, num_frames, dim)  # [1, T, D]
        )

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)
        self.pool = "mean"
        self.to_latent = nn.Identity()
        self.linear_head = nn.Linear(dim, num_classes)

    def forward(self, inputs: dict):
        device = next(self.parameters()).device
        batch, time = next(iter(inputs.values())).shape[:2]
        token_list = []

        for name in self.modalities:
            x = self.embedders[name](inputs[name])  # [B, T, N_patch, D]
            token_list.append(x)

        x = torch.cat(token_list, dim=2)  # [B, T, total_tokens, D]
        pos = self.pos_embedding.to(device, dtype=x.dtype)[None, None, :, :]
        x = x + pos

        x = rearrange(x, "b t n d -> b (t n) d")
        x = x + self.temporal_embedding.repeat_interleave(self.total_tokens, dim=1)[:, :x.shape[1], :]

        x = self.transformer(x)  # [B, T*N, D]
        x = x.mean(dim=1)
        x = self.to_latent(x)
        return self.linear_head(x)
    
if __name__ == "__main__":
    # Example usage
    model = MultiModalTemporalViT(
        image_size=224,
        patch_size=16,
        num_classes=10,
        dim=256,
        depth=4,
        heads=8,
        mlp_dim=512,
        channels_dict={"rgb": 3, "flow": 2, "mask": 1, "kpt": 1},
        dim_head=64,
        num_frames=8
    )
    
    inputs = {
        "rgb": torch.randn(2, 8, 3, 224, 224),
        "flow": torch.randn(2, 8, 2, 224, 224),
        "mask": torch.randn(2, 8, 1, 224, 224),
        "kpt": torch.randn(2, 8, 1, 224, 224)
    }
    
    outputs = model(inputs)
    print(outputs.shape)  # Should be [2, num_classes]