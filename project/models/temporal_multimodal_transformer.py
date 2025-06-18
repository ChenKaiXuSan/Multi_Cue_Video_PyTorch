#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/project/models/temporal_multimodal_transformer.py
Project: /workspace/code/project/models
Created Date: Tuesday June 17th 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Tuesday June 17th 2025 11:40:58 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""
import torch
import torch.nn as nn


class TemporalMultimodalTransformer(nn.Module):
    def __init__(self, d_model=256, n_frames=32, num_classes=10):
        super().__init__()
        self.d_model = d_model

        # 多模态融合（你可以用 Transformer/Gate）
        self.fusion = nn.Linear(4 * d_model, d_model)

        # CLS token & position
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.randn(1, n_frames + 1, d_model))

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)

        # 分类头
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, rgb, mask, flow, kpt):
        # 假设输入都是 [B, T, D]
        fused = torch.cat([rgb, mask, flow, kpt], dim=-1)  # [B, T, 4D]
        x = self.fusion(fused)  # [B, T, D]

        # 加入CLS和位置编码
        B, T, _ = x.shape
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
        x = torch.cat((cls_tokens, x), dim=1)  # [B, T+1, D]
        x = x + self.pos_embed[:, : T + 1]

        # Transformer
        x = x.permute(1, 0, 2)  # [T+1, B, D]
        x = self.encoder(x)
        x = x.permute(1, 0, 2)  # [B, T+1, D]

        # 分类输出
        cls_out = x[:, 0]  # [B, D]
        return self.classifier(cls_out)
