#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/Skiing_Analysis_PyTorch/preprocess/yolov8.py
Project: /workspace/code/Skiing_Analysis_PyTorch/preprocess
Created Date: Wednesday April 23rd 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Thursday April 24th 2025 4:30:44 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn.functional as F

import logging
from ultralytics import YOLO

from utils.utils import process_none

logger = logging.getLogger(__name__)


class YOLOv11Mask:
    def __init__(self, configs) -> None:
        self.yolo_mask = YOLO(configs.YOLO.seg_ckpt)
        self.tracking = configs.YOLO.tracking
        self.conf = configs.YOLO.conf
        self.iou = configs.YOLO.iou
        self.verbose = configs.YOLO.verbose
        self.device = f"cuda:{configs.device}"
        self.save = configs.YOLO.save
        self.save_path = Path(configs.multi_dataset.save_path)
        self.batch_size = configs.batch_size

    def _run_yolo(self, frames_bgr):
        if self.tracking:
            return self.yolo_mask.track(
                source=frames_bgr,
                conf=self.conf,
                iou=self.iou,
                classes=0,
                verbose=self.verbose,
                device=self.device,
            )
        else:
            return self.yolo_mask(
                source=frames_bgr,
                conf=self.conf,
                iou=self.iou,
                classes=0,
                verbose=self.verbose,
                device=self.device,
            )

    def get_YOLO_mask_result(self, vframes: torch.Tensor):
        frames_bgr = vframes.numpy()[:, :, :, ::-1]
        frame_list_bgr = [img for img in frames_bgr]
        return self._run_yolo(frame_list_bgr)

    def resize_masks_to_original(
        self, mask: torch.Tensor, orig_shape: tuple
    ) -> torch.Tensor:
        return F.interpolate(
            mask[None, None], size=orig_shape, mode="bilinear", align_corners=False
        ).squeeze(1)

    def __call__(self, vframes: torch.Tensor, video_path: Path):
        video_name = video_path.stem
        person = video_path.parts[-2]

        save_path = self.save_path / "vis" / "img" / "mask" / person / video_name
        save_crop_path = (
            self.save_path / "vis" / "img" / "mask_crop" / person / video_name
        )
        save_path.mkdir(parents=True, exist_ok=True)
        save_crop_path.mkdir(parents=True, exist_ok=True)

        T = vframes.shape[0]
        none_index, mask_dict, bbox_dict, results = [], {}, {}, []

        for start in range(0, T, self.batch_size):
            batch = vframes[start : min(start + self.batch_size, T)]
            results.extend(self.get_YOLO_mask_result(batch))

        for idx, r in tqdm(enumerate(results), total=T, desc="YOLO Mask", leave=False):
            if r.masks is None:
                none_index.append(idx)
                bbox_dict[idx] = None
                mask_dict[idx] = None
            else:
                mask = (
                    r.masks.data[0]
                    if r.masks.data.ndim == 3
                    else r.masks.data.squeeze(0)
                )
                mask_dict[idx] = self.resize_masks_to_original(mask, r.masks.orig_shape)
                bbox_dict[idx] = r.boxes.xywh[0]

            if self.save:
                r.save(filename=str(save_path / f"{idx}_mask.png"))
                r.save_crop(
                    save_dir=str(save_crop_path), file_name=f"{idx}_mask_crop.png"
                )

        if none_index:
            logger.warning(f"{video_name} has {len(none_index)} frames without bbox.")
            mask_dict = process_none(mask_dict, none_index)

        mask = torch.stack([mask_dict[k] for k in sorted(mask_dict.keys())], dim=0)
        return mask, none_index, results
