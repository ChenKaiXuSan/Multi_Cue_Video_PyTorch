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

import torch
import torch.nn as nn

# multi-modal
from project.models.multi.multimodal_temporal_simple_vit import MultiModalTemporalViT
from project.models.multi.multimodal_res3dcnn import MultiModalRes3DCNN

# single-modal
from project.models.single.temporal_simple_vit import SingleModalTemporalViT
from project.models.single.res3dcnn import Res3DCNN


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
            model = MultiModalTemporalViT(
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
            # FIXME: the single modal vit is not implemented yet.
            model = SingleModalTemporalViT(
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
    elif model_backbone == "3dcnn":
        if modal_type == "all":
            model = MultiModalRes3DCNN(
                class_num= hparams.Res3DCNN.num_classes,
                fuse_method=hparams.Res3DCNN.fuse_method,
                channels_dict=hparams.Res3DCNN.channels_dict,
            )
        elif (
            modal_type == "rgb"
            or modal_type == "flow"
            or modal_type == "mask"
            or modal_type == "kpt"
        ):
            # FIXME: the single modal 3dcnn is not implemented yet.
            model = Res3DCNN(
                class_num=hparams.Res3DCNN.num_classes,
                modal_type=modal_type,
                channels_dict=hparams.Res3DCNN.channels_dict,
            )
    else:
        raise ValueError(f"the modal type {modal_type} is not supported.")

    return model
