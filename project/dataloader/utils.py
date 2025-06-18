#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/project/dataloader/utils.py
Project: /workspace/code/project/dataloader
Created Date: Wednesday April 23rd 2025
Author: Kaixu Chen
-----
Comment:

Copy from pytorchvideo.

Have a good code time :)
-----
Last Modified: Wednesday June 11th 2025 8:56:45 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

from typing import Any, Callable, Dict, Optional

import torch
from PIL import Image
import os

from torchvision.transforms.v2 import functional as F, Transform

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

class UniformTemporalSubsample(Transform):
    """Uniformly subsample ``num_samples`` indices from the temporal dimension of the video.

    Videos are expected to be of shape ``[..., T, C, H, W]`` where ``T`` denotes the temporal dimension.

    When ``num_samples`` is larger than the size of temporal dimension of the video, it
    will sample frames based on nearest neighbor interpolation.

    Args:
        num_samples (int): The number of equispaced samples to be selected
    """

    _transformed_types = (torch.Tensor,)

    def __init__(self, num_samples: int):
        super().__init__()
        self.num_samples = num_samples

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        # inpt = inpt.permute(1, 0, 2, 3)  # [C, T, H, W] -> [T, C, H, W]
        return self._call_kernel(F.uniform_temporal_subsample, inpt, self.num_samples)


class ApplyTransformToKey:
    """
    Applies transform to key of dictionary input.

    Args:
        key (str): the dictionary key the transform is applied to
        transform (callable): the transform that is applied

    Example:
        >>>   transforms.ApplyTransformToKey(
        >>>       key='video',
        >>>       transform=UniformTemporalSubsample(num_video_samples),
        >>>   )
    """

    def __init__(self, key: str, transform: Callable):
        self._key = key
        self._transform = transform

    def __call__(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x[self._key] = self._transform(x[self._key])
        return x


class Div255(torch.nn.Module):
    """
    ``nn.Module`` wrapper for ``pytorchvideo.transforms.functional.div_255``.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Scale clip frames from [0, 255] to [0, 1].
        Args:
            x (Tensor): A tensor of the clip's RGB frames with shape:
                (C, T, H, W).
        Returns:
            x (Tensor): Scaled tensor by dividing 255.
        """
        return x / 255.0


def kpt_to_heatmap(keypoints, H, W, sigma=2):
    """
    kpts: Tensor [B, K, 2] (x, y)
    H, W: 输出热图的高宽
    sigma: 高斯标准差（控制热图的“扩散程度”）
    return: Tensor [B, K, H, W]
    """
    B, K, _ = keypoints.shape
    device = keypoints.device

    # 创建网格
    y = torch.arange(0, H, device=device).view(1, 1, H, 1)
    x = torch.arange(0, W, device=device).view(1, 1, 1, W)

    # 提取关键点坐标
    x_kpt = keypoints[..., 0].view(B, K, 1, 1)
    y_kpt = keypoints[..., 1].view(B, K, 1, 1)

    # 计算高斯响应
    heatmap = torch.exp(-((x - x_kpt) ** 2 + (y - y_kpt) ** 2) / (2 * sigma**2))

    # 可选归一化（最大值归一为1）
    heatmap = heatmap / heatmap.amax(dim=(2, 3), keepdim=True).clamp(min=1e-6)

    heatmap = torch.mean(heatmap, dim=1, keepdim=True)  # [B, 1, H, W]

    return heatmap  # [B, 1, H, W]

def save_sample(tensor: torch.Tensor, out_dir: str, prefix="sample", cmap_name=None, normalize=True):
    """
    Save a [B, C, T, H, W] tensor as image sequences.

    Args:
        tensor: torch.Tensor of shape [B, C, T, H, W]
        out_dir: Directory to save the image sequence
        prefix: Filename prefix
        cmap_name: Optional colormap for single-channel (e.g. 'jet')
        normalize: Whether to scale to [0, 1]
    """
    os.makedirs(out_dir, exist_ok=True)
    
    b, c, t, h, w = tensor.shape
    tensor = tensor[0]  # only visualize first sample

    for ti in range(t):
        img = tensor[:, ti, :, :]  # [C, H, W]

        if c == 3:  # RGB
            img_np = img.permute(1, 2, 0).cpu().clamp(0, 1).numpy()
        elif c == 2:  # optical flow, use magnitude or flow-to-color
            flow_np = img.cpu().numpy()
            magnitude = np.linalg.norm(flow_np, axis=0)
            if normalize:
                magnitude = (magnitude - magnitude.min()) / (magnitude.ptp() + 1e-5)
            img_np = plt.get_cmap("viridis")(magnitude)[:, :, :3]  # RGB heatmap
        elif c == 1:  # mask or heatmap
            img_2d = img[0].cpu().numpy()
            if normalize:
                img_2d = (img_2d - img_2d.min()) / (img_2d.ptp() + 1e-5)
            cmap = get_cmap(cmap_name or "jet")
            img_np = cmap(img_2d)[:, :, :3]  # RGBA → RGB
        else:
            raise ValueError(f"Unsupported channel size: {c}")

        img_uint8 = (img_np * 255).astype(np.uint8)
        Image.fromarray(img_uint8).save(os.path.join(out_dir, f"{prefix}_{ti:03d}.png"))