#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/skeleton/project/models/make_model.py
Project: /workspace/skeleton/project/models
Created Date: Thursday October 19th 2023
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Saturday April 19th 2025 7:58:58 am
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2023 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------

26-11-2024	Kaixu Chen	remove x3d network.
"""

from typing import Any, List

import torch
import torch.nn as nn


# multi-modal
from project.models.multi.multimodal_temporal_simple_vit import MultiModalTemporalViT

# single-modal
from project.models.single.temporal_simple_vit import SingleModalTemporalViT

# from project.models.single.vivit import ViViT


class MakeVideoModule(nn.Module):
    """
    make 3D CNN model from the PytorchVideo lib.

    """

    def __init__(self, hparams) -> None:
        super().__init__()

        self.model_name = hparams.model.backbone
        self.model_class_num = hparams.model.model_class_num
        self.model_depth = hparams.model.model_depth
        self.model = self.initialize_walk_resnet(self.model_class_num)

        self.fuse_method = hparams.model.fuse_method

    def initialize_walk_resnet(self, input_channel: int = 3) -> nn.Module:
        slow = torch.hub.load(
            "facebookresearch/pytorchvideo", "slow_r50", pretrained=True
        )

        # for the folw model and rgb model
        slow.blocks[0].conv = nn.Conv3d(
            input_channel,
            64,
            kernel_size=(1, 7, 7),
            stride=(1, 2, 2),
            padding=(0, 3, 3),
            bias=False,
        )
        # change the knetics-400 output 400 to model class num
        slow.blocks[-1].proj = nn.Linear(2048, self.model_class_num)

        return slow

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        if self.model_name == "3dcnn":
            return self.initialize_walk_resnet()
        else:
            raise KeyError(f"the model name {self.model_name} is not in the model zoo")

    def forward(self, video: torch.Tensor, attn_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video: (B, C, T, H, W)
            attn_map: (B, 1, T, H, W)

        Returns:
            torch.Tensor: (B, C, T, H, W)
        """
        # video = self.video_cnn(video)
        # attn_map = self.attn_map(attn_map)

        # return video * attn_map
        return video


class MakeImageModule(nn.Module):
    """
    the module zoo from the torchvision lib, to make the different 2D model.

    """

    def __init__(self, hparams) -> None:
        super().__init__()

        self.model_name = hparams.model.model
        self.model_class_num = hparams.model.model_class_num
        self.transfer_learning = hparams.train.transfer_learning

    def make_resnet(self, input_channel: int = 3) -> nn.Module:
        if self.transfer_learning:
            model = torch.hub.load(
                "pytorch/vision:v0.10.0", "resnet50", pretrained=True
            )
            model.conv1 = nn.Conv2d(
                input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            model.fc = nn.Linear(2048, self.model_class_num)

        return model

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        if self.model_name == "resnet":
            return self.make_resnet()
        else:
            raise KeyError(f"the model name {self.model_name} is not in the model zoo")


def select_model(hparams) -> nn.Module:
    """
    Select the model based on the hparams.

    Args:
        hparams: the hyperparameters of the model.

    Returns:
        nn.Module: the selected model.
    """

    model_backbone = hparams.model.backbone
    modal_type = hparams.train.modal_type

    if model_backbone == "vit":
        if modal_type == "all":
            return MultiModalTemporalViT(
                image_size=hparams.Vit.image_size,
                patch_size=hparams.Vit.patch_size,
                num_classes=hparams.Vit.num_classes,
                dim=hparams.Vit.dim,
                depth=hparams.Vit.depth,
                heads=hparams.Vit.heads,
                mlp_dim=hparams.Vit.mlp_dim,
                channels_dict=hparams.Vit.channel_dict,
                dim_head=hparams.Vit.dim_head,
                num_frames=hparams.Vit.num_frames,
            )
        elif (
            modal_type == "rgb"
            or modal_type == "flow"
            or modal_type == "mask"
            or modal_type == "kpt"
        ):
            return SingleModalTemporalViT(
                image_size=hparams.Vit.image_size,
                patch_size=hparams.Vit.patch_size,
                num_classes=hparams.Vit.num_classes,
                dim=hparams.Vit.dim,
                depth=hparams.Vit.depth,
                heads=hparams.Vit.heads,
                mlp_dim=hparams.Vit.mlp_dim,
                channels_dict=hparams.Vit.channel_dict,
                dim_head=hparams.Vit.dim_head,
                num_frames=hparams.Vit.num_frames,
            )
        else:
            raise ValueError(f"the modal type {modal_type} is not supported.")
