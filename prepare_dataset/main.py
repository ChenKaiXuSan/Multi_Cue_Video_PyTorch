#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/skeleton/prepare_gait_cycle_index/main_yolov8.py
Project: /workspace/skeleton/prepare_gait_cycle_index
Created Date: Thursday June 13th 2024
Author: Kaixu Chen
-----
Comment:
This script used for STGCN, a skeleton-based action recognition model.
The script will load the video from the dataset, and use the yolov8 to get the keypoint, save into .pkl file.

Have a good code time :)
-----
Last Modified: Thursday June 13th 2024 1:21:41 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2024 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

from __future__ import annotations

import torch
from pathlib import Path
from torchvision.io import read_video

import multiprocessing
import logging
import time

import hydra

from prepare_dataset.preprocess import Preprocess
from utils.utils import timing, save_to_pt_gz

RAW_CLASS = ["ASD", "DHS", "LCS", "HipOA"]
CLASS = ["ASD", "DHS", "LCS_HipOA", "Normal"]
map_CLASS = {"ASD": 0, "DHS": 1, "LCS_HipOA": 2, "Normal": 3}  # map the class to int

logger = logging.getLogger(__name__)


class LoadOneDisease:
    def __init__(self, data_path: str | Path, fold: str, diseases: list[str]) -> None:
        self.DATA_PATH = Path(data_path)
        self.fold = fold
        self.diseases = diseases
        self.path_dict = {key: [] for key in CLASS}

    def process_class(self, dir_path: Path, flag: str):
        for video_file in sorted(dir_path.iterdir()):
            for disease in self.diseases:
                if disease in video_file.name.split("_"):
                    info = {"flag": flag, "disease": disease}
                    key = (
                        disease
                        if disease in CLASS
                        else CLASS[2]  # default to LCS_HipOA
                    )
                    self.path_dict[key].append((video_file, info))
                    break  # avoid double append if multiple diseases match

    def __call__(self) -> dict:
        fold_path = self.DATA_PATH / self.fold

        for flag in ["train", "val"]:
            flag_path = fold_path / flag
            for disease in self.diseases:
                subdir = "ASD_not" if disease in ["DHS", "LCS", "HipOA"] else disease
                target_dir = flag_path / subdir
                if target_dir.exists():
                    self.process_class(target_dir, flag)
                else:
                    logging.warning(f"Directory not found: {target_dir}")

        # sort the path_dict by disease
        for key in self.path_dict:
            self.path_dict[key].sort(key=lambda x: x[0].name)

        return self.path_dict


@timing(logger=logger)
def process(parames, fold: str, disease: list):
    DATA_PATH = Path(parames.multi_dataset.data_path)
    SAVE_PATH = Path(parames.multi_dataset.save_path)

    # prepare the log file
    logger = logging.getLogger(f"Logger-{multiprocessing.current_process().name}")

    logger.info(f"Start process the {fold} dataset")

    # prepare the preprocess
    preprocess = Preprocess(parames)

    # * step1: load the video path, with the sorted order
    load_one_disease = LoadOneDisease(DATA_PATH, fold, disease)
    one_fold_video_path_dict = (
        load_one_disease()
    )  # {disease: (video_path, info{flag, disease}})}

    # k is disease, v is (video_path, info)
    for k, v in one_fold_video_path_dict.items():
        if v:
            logger.info(f"Start process the {k} disease")

        for video_path, info in v:
            # * step2: load the video from vieo path
            # get the bbox
            vframes, audio, _ = read_video(
                video_path, pts_unit="sec", output_format="THWC"
            )

            # * step3: use preprocess to get information.
            # the format is: final_frames, bbox_none_index, label, optical_flow, bbox, mask, pose
            label = torch.tensor([map_CLASS[k]])  # convert the label to int

            (
                bbox_none_index,
                optical_flow,
                bbox,
                mask,
                keypoints,
                keypoints_score,
            ) = preprocess(vframes, video_path)

            # * step4: save the video frames keypoint
            anno = dict()

            # * when use mmaction, we need convert the keypoint torch to numpy
            anno["video"] = video_path
            anno["frames"] = vframes.permute(0, 3, 1, 2).cpu()  # (T, C, H, W)
            anno["label"] = int(label)
            anno["total_frames"] = vframes.shape[0]
            anno["img_shape"] = (vframes.shape[1], vframes.shape[2])
            anno["bbox_none_index"] = bbox_none_index

            # multi modiality
            anno["optical_flow"] = optical_flow.cpu()
            anno["bbox"] = bbox.cpu()
            anno["mask"] = mask.cpu()
            anno["keypoint"] = keypoints.cpu()
            anno["keypoint_score"] = keypoints_score.cpu()

            save_to_pt_gz(video_path, SAVE_PATH, anno)

            del anno
            del optical_flow, bbox, mask, keypoints, keypoints_score
            torch.cuda.empty_cache()

def batched(iterable, n):
    for i in range(0, len(iterable), n):
        yield iterable[i : i + n]


@hydra.main(config_path="../configs/", config_name="prepare_dataset")
def main(parames):
    """
    main, for the multiprocessing using.

    Args:
        parames (hydra): hydra config.
    """

    # ! only for test
    # process(parames, "fold0", ["ASD"])

    for disease in [["ASD"], ["LCS", "HipOA"], ["DHS"]]:
        logger.info(f"Start process for {disease}")
        process(parames, "fold0", disease)

    # FIXME: 不知道为什么用不了多线程，一条会莫名的死掉
    # task_config = [
    #     ("0", "fold0", ["ASD"]),
    #     ("1", "fold0", ["LCS", "HipOA"]),
    #     ("0", "fold0", ["DHS"]),
    # ]

    # for batch in batched(task_config, 2):
    #     logger.info(f"Start batch process for {batch}")

    #     threads = []

    #     for device, fold, disease in batch:
    #         parames.device = device
    #         thread = multiprocessing.Process(target=process, args=(parames, fold, disease))
    #         threads.append(thread)

    #     # start all threads
    #     for thread in threads:
    #         logger.info(f"Start thread {thread.name} for {thread._target.__name__}")
    #         thread.start()

    #     # wait for all threads to finish
    #     for thread in threads:
    #         thread.join()

    #     time.sleep(1)  # sleep for a while to avoid too many processes at the same time

    logger.info("All processes finished.")


if __name__ == "__main__":
    # torch.multiprocessing.set_start_method('spawn')# good solution !!!!
    main()
