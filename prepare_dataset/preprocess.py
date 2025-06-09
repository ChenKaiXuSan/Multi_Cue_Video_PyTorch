#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/Skiing_Analysis_PyTorch/prepare_dataset/preprocess.py
Project: /workspace/code/Skiing_Analysis_PyTorch/prepare_dataset
Created Date: Wednesday April 23rd 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Wednesday April 23rd 2025 12:56:44 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

import logging
from pathlib import Path

import torch

from prepare_dataset.optical_flow import OpticalFlow
from prepare_dataset.yolov11_bbox import YOLOv11Bbox
from prepare_dataset.yolov11_pose import YOLOv11Pose
from prepare_dataset.yolov11_mask import YOLOv11Mask

logger = logging.getLogger(__name__)


class Preprocess:
    def __init__(self, config) -> None:
        super(Preprocess, self).__init__()

        self.config = config
        self.task = config.task
        logger.info(f"Preprocess task: {self.task}")

        # 模块初始化
        self.yolo_model_bbox = self._init_task("bbox", YOLOv11Bbox)
        self.yolo_model_pose = self._init_task("pose", YOLOv11Pose)
        self.yolo_model_mask = self._init_task("mask", YOLOv11Mask)
        self.of_model = self._init_task("optical_flow", OpticalFlow)

    def _init_task(self, task_name: str, cls):
        return cls(self.config) if task_name in self.task else None

    def _empty_tensor(self, shape: tuple, dtype=torch.float32):
        """
        Create an empty tensor with the given shape and dtype.
        """
        return torch.empty(shape, dtype=dtype)

    def shape_check(self, check_dict: dict[str, torch.Tensor]):
        """
        shape_check check the given value shape, and assert the shape.

        check list include:
        # batch, (t, h, w, c)
        # bbox, (t, 4) (cxcywh)
        # mask, (1, t, h, w)
        # keypoint, (t, 17, 2)
        # optical_flow, (2, t, h, w)

        Args:
            check (list): checked value, in list.
        """

        t, h, w, c = check_dict["video"].shape

        for key, ck in check_dict.items():

            if ck is None or key == "video":
                continue

            elif key == "optical_flow":
                # optical flow shape check
                assert ck.shape == (0, 2, h, w) or ck.shape == (t, 2, h, w), (
                    f"Optical flow shape mismatch: {ck.shape} != {(0, 2, h, w)} or {(t, 2, h, w)}"
                )
            elif key == "bbox":
                # bbox shape check
                assert ck.shape == (0, 4) or ck.shape == (t, 4), (
                    f"Bounding box shape mismatch: {ck.shape} != {(0, 4)} or {(t, 4)}"
                )
            elif key == "mask":
                # mask shape check
                assert ck.shape == (0, 1, h, w) or ck.shape == (t, 1, h, w), (
                    f"Mask shape mismatch: {ck.shape} != {(0, 1, h, w)} or {(t, 1, h, w)}"
                )
            elif key == "pose":
                # pose shape check
                assert ck.shape == (0, 17, 2) or ck.shape == (t, 17, 2), (
                    f"Pose shape mismatch: {ck.shape} != {(0, 17, 2)} or {(t, 17, 2)}"
                )
            elif key == "pose_score":
                # pose score shape check
                assert ck.shape == (0, 17) or ck.shape == (t, 17), (
                    f"Pose score shape mismatch: {ck.shape} != {(0, 17)} or {(t, 17)}"
                )
            else:
                raise ValueError(f"Unknown key in check_dict: {key}")
            
    def __call__(self, vframes: torch.Tensor, video_path: Path):

        # change the video_path to video name
        # TODO: change the disease
        if "LCS" in video_path.stem or "HipOA" in video_path.stem:
            video_path = Path(str(video_path).replace("ASD_not", "LCS_HipOA"))
        elif "DHS" in video_path.stem:
            video_path = Path(str(video_path).replace("ASD_not", "DHS"))

        T, H, W, C = vframes.shape

        # * process optical flow
        if self.of_model:
            optical_flow = self.of_model(vframes, video_path)
        else:
            optical_flow = self._empty_tensor((0, 2, H, W), dtype=torch.float32)

        # * process bbox
        if self.yolo_model_bbox:
            # use MultiPreprocess to process bbox, mask, pose
            bbox, bbox_none_index, bbox_results = self.yolo_model_bbox(
                vframes, video_path
            )
        else:
            bbox_none_index = []
            bbox = self._empty_tensor((0, 4), dtype=torch.float32)

        # * process pose
        if self.yolo_model_pose:
            pose, pose_score, pose_none_index, pose_results = self.yolo_model_pose(
                vframes, video_path
            )
        else:
            pose = self._empty_tensor((0, 17, 3), dtype=torch.float32)
            pose_score = self._empty_tensor((0, 17), dtype=torch.float32)

        # * process mask
        if self.yolo_model_mask:
            mask, mask_none_index, mask_results = self.yolo_model_mask(
                vframes, video_path
            )
        else:
            mask = self._empty_tensor((0, 1, H, W), dtype=torch.float32)

        # * shape check
        self.shape_check(
            {
                "video": vframes,
                "optical_flow": optical_flow,
                "bbox": bbox,
                "mask": mask,
                "pose": pose,
                "pose_score": pose_score,
            }
        )

        return bbox_none_index, optical_flow, bbox, mask, pose, pose_score
