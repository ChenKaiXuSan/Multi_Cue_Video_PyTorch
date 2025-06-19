#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/project/dataloader/gait_video_dataset.py
Project: /workspace/code/project/dataloader
Created Date: Tuesday April 22nd 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Tuesday April 22nd 2025 11:18:09 am
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------

04-05-2025	Kaixu Chen	load the video as batch, this will save the CPU memory.

23-04-2025	Kaixu Chen	init the code.
"""

from __future__ import annotations

import logging

from typing import Any, Callable, Dict, Optional, Tuple

import torch
import gzip

from project.dataloader.utils import kpt_to_heatmap, save_sample

logger = logging.getLogger(__name__)


class LabeledGaitVideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        experiment: str,
        labeled_video_paths: list[Tuple[str, Optional[dict]]],
        transform: Optional[Callable[[dict], Any]] = None,
    ) -> None:
        super().__init__()

        self._transform = transform
        self._labeled_videos = labeled_video_paths
        self._experiment = experiment

    def __len__(self):
        return len(self._labeled_videos)

    def move_transform(self, vframes: torch.Tensor, fps: int) -> torch.Tensor:

        t, *_ = vframes.shape

        batch_res = []

        for f in range(0, t, fps):
            one_sec_vframes = vframes[f : f + fps, :, :, :]  # t, c, h, w

            if self._transform is not None:
                transformed_img = self._transform(one_sec_vframes)
                transformed_img = transformed_img.to(dtype=torch.float32)

                batch_res.append(transformed_img.permute(1, 0, 2, 3))  # c, t, h, w
            else:
                logger.warning("no transform")
                batch_res.append(one_sec_vframes.permute(1, 0, 2, 3))  # c, t, h, w

        return torch.stack(batch_res, dim=0)  # b, c, t, h, w

    def __getitem__(self, index) -> dict[str, Any]:

        with gzip.open(self._labeled_videos[index], "r") as f:
            file_info_dict = torch.load(f)

        # load video info from json file
        video_path = file_info_dict["video"]
        img_shape = file_info_dict["img_shape"]
        bbox_non_index = file_info_dict["bbox_none_index"]
        bbox = file_info_dict["bbox"]
        keypoint_score = file_info_dict["keypoint_score"]

        # multi-modal data
        frames = file_info_dict["frames"]
        label = file_info_dict["label"]
        optical_flow = file_info_dict["optical_flow"]
        mask = file_info_dict["mask"]
        keypoints = file_info_dict["keypoint"]

        keypoints_heatmap = kpt_to_heatmap(
            keypoints=keypoints,
            H=img_shape[0],
            W=img_shape[1],
            sigma=2,
        )

        # transform the video frames
        transformed_vframes = self.move_transform(frames, 30)
        transformed_optical_flow = self.move_transform(optical_flow, 30)
        transformed_keypoints_heatmap = self.move_transform(keypoints_heatmap, 30)
        transformed_mask = self.move_transform(mask, 30)

        sample_info_dict = {
            "label": torch.tensor(label),
            "rgb": transformed_vframes,
            "flow": transformed_optical_flow,
            "kpt_heatmap": transformed_keypoints_heatmap,
            "mask": transformed_mask,
        }

        # visualize the sample image 
        # save_sample(transformed_vframes, "vis/frames", prefix="frame")
        # save_sample(transformed_optical_flow, "vis/optical_flow", prefix="flow")
        # save_sample(transformed_keypoints_heatmap, "vis/keypoints_heatmap", prefix="kpt_heatmap")
        # save_sample(transformed_mask, "vis/mask", prefix="mask")

        return sample_info_dict


def whole_video_dataset(
    experiment: str,
    transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    dataset_idx: list = [],
) -> LabeledGaitVideoDataset:
    dataset = LabeledGaitVideoDataset(
        experiment=experiment,
        transform=transform,
        labeled_video_paths=dataset_idx,
    )

    return dataset
