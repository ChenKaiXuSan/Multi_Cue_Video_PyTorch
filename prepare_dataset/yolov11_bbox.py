#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/prepare_dataset/yolov11 copy.py
Project: /workspace/code/prepare_dataset
Created Date: Tuesday June 3rd 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Tuesday June 3rd 2025 12:40:59 pm
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
import logging

import torch

from ultralytics import YOLO

from utils.utils import process_none

logger = logging.getLogger(__name__)

class YOLOv11Bbox:
    def __init__(self, configs) -> None:
        self.yolo_bbox = YOLO(configs.YOLO.bbox_ckpt)
        self.tracking = configs.YOLO.tracking

        self.conf = configs.YOLO.conf
        self.iou = configs.YOLO.iou
        self.verbose = configs.YOLO.verbose
        self.device = f"cuda:{configs.device}"

        self.batch_size = configs.batch_size
        self.save = configs.YOLO.save
        self.save_path = Path(configs.multi_dataset.save_path)

    def _run_yolo(self, frames_bgr):
        if self.tracking:
            return self.yolo_bbox.track(
                source=frames_bgr,
                conf=self.conf,
                iou=self.iou,
                classes=0,
                verbose=self.verbose,
                device=self.device,
            )
        else:
            return self.yolo_bbox.predict(
                source=frames_bgr,
                conf=self.conf,
                iou=self.iou,
                classes=0,
                verbose=self.verbose,
                device=self.device,
            )

    def get_YOLO_bbox_result(self, vframes: torch.Tensor):
        frames_bgr = vframes.numpy()[:, :, :, ::-1]
        frame_list_bgr = [img for img in frames_bgr]
        return self._run_yolo(frame_list_bgr)

    def __call__(self, vframes: torch.Tensor, video_path: Path):
        video_name = video_path.stem
        person = video_path.parts[-2]

        save_path = self.save_path / "vis" / "img" / "bbox" / person / video_name
        save_crop_path = (
            self.save_path / "vis" / "img" / "bbox_crop" / person / video_name
        )
        save_path.mkdir(parents=True, exist_ok=True)
        save_crop_path.mkdir(parents=True, exist_ok=True)

        T = vframes.shape[0]
        none_index, bbox_dict, results = [], {}, []

        for start in range(0, T, self.batch_size):
            batch = vframes[start : min(start + self.batch_size, T)]
            results.extend(self.get_YOLO_bbox_result(batch))

        for idx, r in tqdm(enumerate(results), total=T, desc="YOLO BBox", leave=False):
            if r.boxes is None or r.boxes.shape[0] == 0:
                none_index.append(idx)
                bbox_dict[idx] = None
            else:
                bbox_dict[idx] = (
                    r.boxes.xywh[0] if r.boxes.xywh.ndim == 2 else r.boxes.xywh
                )

            if self.save:
                r.save(filename=str(save_path / f"{idx}_bbox.png"))
                r.save_crop(
                    save_dir=str(save_crop_path), file_name=f"{idx}_bbox_crop.png"
                )

        if none_index:
            logger.warning(f"{video_name} has {len(none_index)} frames without bbox.")
            bbox_dict = process_none(bbox_dict, none_index)

        bbox = torch.stack([bbox_dict[k] for k in sorted(bbox_dict.keys())], dim=0)
        return bbox, none_index, results
