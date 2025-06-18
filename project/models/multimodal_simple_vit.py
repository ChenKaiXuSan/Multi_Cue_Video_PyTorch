#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
File: /workspace/code/project/models/multimodal_simple_vit.py
Project: /workspace/code/project/models
Created Date: Wednesday June 18th 2025
Author: Kaixu Chen
-----
Comment:

https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/simple_vit.py

Have a good code time :)
-----
Last Modified: Wednesday June 18th 2025 11:25:13 am
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

# --- Helper functions ---

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

# --- Basic Modules ---

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
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head),
                FeedForward(dim, mlp_dim)
            ]))

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

        assert image_height % patch_height == 0 and image_width % patch_width == 0
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.embed = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.num_patches = num_patches

    def forward(self, x):
        return self.embed(x)

# --- Multi-modal ViT ---

class MultiModalViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim,
                 channels_dict, dim_head=64):
        """
        channels_dict: dict, e.g. {"rgb":3, "flow":2, "mask":1, "kpt":1}
        """
        super().__init__()
        self.modalities = list(channels_dict.keys())

        # Embedding for each modality
        self.embedders = nn.ModuleDict({
            name: PatchEmbedding(image_size, patch_size, channels, dim)
            for name, channels in channels_dict.items()
        })

        # Positional embedding
        h, w = pair(image_size)
        ph, pw = pair(patch_size)
        self.num_patches = (h // ph) * (w // pw)
        total_tokens = self.num_patches * len(self.modalities)
        self.pos_embedding = nn.Parameter(
            posemb_sincos_2d(h // ph, w // pw, dim).repeat(len(self.modalities), 1)
        )  # [N_modal * N_patch, D]

        # Transformer
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        # Classification head
        self.pool = "mean"
        self.to_latent = nn.Identity()
        self.linear_head = nn.Linear(dim, num_classes)

    def forward(self, inputs: dict):
        """
        inputs: dict with keys matching self.modalities
        Example:
        {
            "rgb":  [B, 3, H, W],
            "flow": [B, 2, H, W],
            "mask": [B, 1, H, W],
            "kpt":  [B, 1, H, W],
        }
        """
        device = next(self.parameters()).device
        token_list = []

        for i, name in enumerate(self.modalities):
            x = self.embedders[name](inputs[name])  # [B, N_patch, D]
            token_list.append(x)

        x = torch.cat(token_list, dim=1)  # [B, N_modal * N_patch, D]

        pos = self.pos_embedding.to(device, dtype=x.dtype).unsqueeze(0)  # [1, N_tokens, D]
        x = x + pos

        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.to_latent(x)
        return self.linear_head(x)
    

if __name__ == "__main__":
    # Example usage
    model = MultiModalViT(
        image_size=(224, 224),
        patch_size=(16, 16),
        num_classes=2,
        dim=256,
        depth=6,
        heads=8,
        mlp_dim=512,
        channels_dict={"rgb": 3, "flow": 2, "mask": 1, "kpt": 1}
    )
    
    inputs = {
        "rgb": torch.randn(2, 3, 224, 224),
        "flow": torch.randn(2, 2, 224, 224),
        "mask": torch.randn(2, 1, 224, 224),
        "kpt": torch.randn(2, 1, 224, 224)
    }
    
    outputs = model(inputs)
    print(outputs.shape)  # Should be [2, num_classes]