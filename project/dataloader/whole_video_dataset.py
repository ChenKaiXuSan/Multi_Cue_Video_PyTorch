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
import json

from typing import Any, Callable, Dict, Optional, Tuple

import torch

from torchvision.io import read_video

from project.dataloader.med_attn_map import MedAttnMap

logger = logging.getLogger(__name__)


class LabeledGaitVideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        experiment: str,
        labeled_video_paths: list[Tuple[str, Optional[dict]]],
        transform: Optional[Callable[[dict], Any]] = None,
        doctor_res_path: str = "",
        skeleton_path: str = "",
    ) -> None:
        super().__init__()

        self._transform = transform
        self._labeled_videos = labeled_video_paths
        # self._index_map = self.prepare_video_mapping_info(
        #     labeled_video_paths=labeled_video_paths,
        #     clip_duration=clip_duration,
        # )
        self._experiment = experiment

        if "True" in self._experiment:
            self.attn_map = MedAttnMap(doctor_res_path, skeleton_path)

    def __len__(self):
        return len(self._labeled_videos)

    def move_transform(self, vframes: torch.Tensor, fps: int) -> torch.Tensor:

        t, *_ = vframes.shape

        batch_res = []

        for f in range(0, t, fps):
            one_sec_vframes = vframes[f : f + fps, :, :, :]

            if self._transform is not None:
                transformed_img = self._transform(one_sec_vframes.permute(1, 0, 2, 3))

                batch_res.append(transformed_img.permute(1, 0, 2, 3))  # c, t, h, w
            else:
                logger.warning("no transform")
                batch_res.append(one_sec_vframes.permute(1, 0, 2, 3))  # c, t, h, w

        return torch.stack(batch_res, dim=0)  # b, c, t, h, w

    def __getitem__(self, index) -> dict[str, Any]:

        with open(self._labeled_videos[index]) as f:
            file_info_dict = json.load(f)

        # load video info from json file
        video_name = file_info_dict["video_name"]
        video_path = file_info_dict["video_path"]

        vframes, _, info = read_video(video_path, output_format="TCHW", pts_unit="sec")

        label = file_info_dict["label"]
        disease = file_info_dict["disease"]
        # gait_cycle_index = file_info_dict["gait_cycle_index"]
        # bbox_none_index = file_info_dict["none_index"]
        # bbox = file_info_dict["bbox"]

        attn_map = self.attn_map(
            video_name=video_name,
            video_path=video_path,
            disease=disease,
            vframes=vframes,
        )

        # transform the video frames
        transformed_vframes = self.move_transform(vframes, int(info["video_fps"]))
        transformed_attn_map = self.move_transform(attn_map, int(info["video_fps"]))

        sample_info_dict = {
            "video": transformed_vframes,
            "label": label,
            "attn_map": transformed_attn_map,
            "disease": disease,
            "video_name": video_name,
            "video_index": index,
            # "bbox_none_index": bbox_none_index,
        }

        # logger.info(f"the video name is {video_name}")
        # logger.info(f"the batch size is {transformed_vframes.shape}")

        return sample_info_dict


def whole_video_dataset(
    experiment: str,
    transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    dataset_idx: list = [],
    doctor_res_path: str = "",
    skeleton_path: str = "",
    clip_duration: int = 1,
) -> LabeledGaitVideoDataset:
    dataset = LabeledGaitVideoDataset(
        experiment=experiment,
        transform=transform,
        labeled_video_paths=dataset_idx,
        doctor_res_path=doctor_res_path,
        skeleton_path=skeleton_path,
    )

    return dataset
