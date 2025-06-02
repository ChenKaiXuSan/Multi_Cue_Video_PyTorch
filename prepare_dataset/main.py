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

import pickle
import torch
from pathlib import Path
from torchvision.io import read_video

import multiprocessing
import logging

import hydra

from utils import del_folder, make_folder, merge_pkl
from yolov8 import MultiPreprocess
from preprocess import Preprocess

RAW_CLASS = ["ASD", "DHS", "LCS", "HipOA"]
CLASS = ["ASD", "DHS", "LCS_HipOA", "Normal"]
map_CLASS = {"ASD": 0, "DHS": 1, "LCS_HipOA": 2, "Normal": 3}  # map the class to int


class LoadOneDisese:
    def __init__(self, data_path, fold, disease) -> None:

        self.DATA_PATH = data_path
        self.fold = fold
        self.disease = disease
        # this is the return dict
        self.path_dict = {key: [] for key in CLASS}

    def process_class(self, one_class: Path, flag: str):
        for d in self.disease:
            if d in ["DHS", "LCS", "HipOA"]:
                for one_video in sorted(one_class.iterdir()):
                    if d in one_video.name.split("_"):
                        info = {
                            "flag": flag,
                            "disease": d,
                        }

                        if d in CLASS:
                            self.path_dict[d].append((str(one_video), info))
                        else:
                            self.path_dict[CLASS[2]].append((str(one_video), info))
                    else:
                        continue

            else:
                for one_video in sorted(one_class.iterdir()):
                    if d in one_video.name.split("_"):
                        info = {"flag": flag, "disease": d}

                        self.path_dict[d].append((str(one_video), info))

    def __call__(self) -> dict:
        one_fold_path = Path(self.DATA_PATH, self.fold)

        for flag in ["train", "val"]:
            one_flag_path = Path(one_fold_path, flag)

            for one_class in self.disease:
                if one_class == "DHS" or one_class == "LCS" or one_class == "HipOA":
                    new_path = one_flag_path / "ASD_not"
                else:
                    new_path = one_flag_path / one_class

                self.process_class(new_path, flag)

        return self.path_dict


def process(parames, fold: str, disease: list):
    DATA_PATH = parames.gait_dataset.data_path
    SAVE_PATH = Path(parames.gait_dataset.save_path)

    res = {"train": [], "val": []}

    # prepare the log file
    logger = logging.getLogger(f"Logger-{multiprocessing.current_process().name}")

    logger.info(f"Start process the {fold} dataset")

    # prepare the preprocess
    preprocess = Preprocess(parames)

    # * step1: load the video path, with the sorted order
    load_one_disease = LoadOneDisese(DATA_PATH, fold, disease)
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
            # THWC to CTHW
            m_vframes = vframes.permute(3, 0, 1, 2)
            (
                frames,
                bbox_none_index,
                label,
                optical_flow,
                bbox,
                mask,
                keypoints,
                keypoints_score,
            ) = preprocess(m_vframes, label, 0)

            # * step4: save the video frames keypoint
            anno = dict()

            # * when use mmaction, we need convert the keypoint torch to numpy
            anno["video_dir"] = video_path
            anno["frames"] = vframes.cpu()
            anno["label"] = int(label)
            anno["total_frames"] = vframes.shape[0]
            anno["img_shape"] = (vframes.shape[1], vframes.shape[2])
            
            # multi modiality
            anno["optical_flow"] = optical_flow.cpu()
            anno["bbox"] = bbox.cpu()
            anno["mask"] = mask.cpu()
            anno["keypoint"] = keypoints.cpu()
            anno["keypoint_score"] = keypoints_score.cpu()

            res[info["flag"]].append(anno)

            # break;

    # save one disease to pkl file.
    SAVE_PATH.mkdir(parents=True, exist_ok=True)

    with open(SAVE_PATH / f'{"_".join(disease)}.pkl', "wb") as file:
        pickle.dump(res, file)

    logging.info(f"Save the {fold} {disease} to {SAVE_PATH}")




@hydra.main(
    config_path="../configs/", config_name="prepare_dataset"
)
def main(parames):
    """
    main, for the multiprocessing using.

    Args:
        parames (hydra): hydra config.
    """

    # ! only for test
    process(parames, "fold0", ["ASD"])

    # threads = []
    # for d in [["ASD"], ["LCS", "HipOA"]]:

    #     thread = multiprocessing.Process(target=process, args=(parames, "fold0", d))
    #     threads.append(thread)

    # for t in threads:
    #     t.start()

    # for t in threads:
    #     t.join()

    # process(parames, "fold0", ["DHS"])

    # merge_pkl(parames)


if __name__ == "__main__":
    # torch.multiprocessing.set_start_method('spawn')# good solution !!!!
    main()
