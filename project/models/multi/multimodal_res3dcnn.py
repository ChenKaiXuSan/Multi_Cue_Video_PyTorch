#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/project/models/make_model copy.py
Project: /workspace/code/project/models
Created Date: Thursday May 8th 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Thursday May 8th 2025 1:23:28 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date          By      Comments
----------    ---     ---------------------------------------------------------
"""

import torch
import torch.nn as nn
from typing import Dict

class MultiModalRes3DCNN(nn.Module):
    def __init__(self, class_num: int, fuse_method: str, channels_dict: Dict[str, int]) -> None:
        super().__init__()
        self.class_num = class_num
        self.fuse_method = fuse_method
        self.channels_dict = channels_dict

        self.total_in_channels = sum(self.channels_dict.values())

        self.model = torch.hub.load(
            "facebookresearch/pytorchvideo", "slow_r50", pretrained=True
        )

        self.model.blocks[0].conv = nn.Conv3d(
            in_channels=self.total_in_channels,
            out_channels=64,
            kernel_size=(1, 7, 7),
            stride=(1, 2, 2),
            padding=(0, 3, 3),
            bias=False,
        )
        self.model.blocks[-1].proj = nn.Linear(2048, self.class_num)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        expected_modalities = list(self.channels_dict.keys())

        for mod in expected_modalities:
            if mod not in inputs:
                raise ValueError(f"Missing modality: {mod}")

        tensors = [inputs[mod] for mod in expected_modalities]

        if self.fuse_method == "concat":
            x = torch.cat(tensors, dim=1)
        elif self.fuse_method == "add":
            x = torch.stack(tensors, dim=0).sum(dim=0)
        elif self.fuse_method == "mul":
            x = tensors[0]
            for t in tensors[1:]:
                x = x * t
        elif self.fuse_method == "none":
            x = tensors[0]
        else:
            raise ValueError(f"Unsupported fuse method: {self.fuse_method}")

        return self.model(x)

if __name__ == "__main__":

    model = MultiModalRes3DCNN(
        class_num=3,
        fuse_method='concat',
        channels_dict={'rgb': 3, 'flow': 2, 'kpt': 1, 'mask': 1}    
    )

    # b, c, t, h, w
    inputs = { 
        "rgb": torch.randn(2, 3, 8, 224, 224),
        "flow": torch.randn(2, 2, 8, 224, 224),
        "kpt": torch.randn(2, 1, 8, 224, 224),
        "mask": torch.randn(2, 1, 8, 224, 224)
    }

    out = model(inputs)
    print(out.shape)  # Expected: [2, 3]