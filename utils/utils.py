#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/utils/utils copy.py
Project: /workspace/code/utils
Created Date: Saturday June 7th 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Saturday June 7th 2025 4:25:18 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

from typing import Dict, List
import copy

import time
import logging
from functools import wraps
import gzip

import os
import shutil

import cv2
from tqdm import tqdm

from pathlib import Path

import torch
from torchvision.transforms.functional import crop, pad, resize

import logging

logger = logging.getLogger(__name__)


def clip_pad_with_bbox(
    imgs: torch.tensor, boxes: list, img_size: int = 256, bias: int = 10
):
    """
    based torchvision function to crop, pad, resize img.

    clip with the bbox, (x1-bias, y1) and padd with the (gap-bais) in left and right.

    Args:
        imgs (list): imgs with (h, w, c)
        boxes (list): (x1, y1, x2, y2)
        img_size (int, optional): croped img size. Defaults to 256.
        bias (int, optional): the bias of bbox, with the (x1-bias) and (x2+bias). Defaults to 5.

    Returns:
        tensor: (c, t, h, w)
    """
    object_list = []

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)  # dtype must int for resize, crop function

        box_width = x2 - x1
        box_height = y2 - y1

        width_gap = int(((box_height - box_width) / 2))  # keep int type

        img = imgs  # (h, w, c) to (c, h, w), for pytorch function

        # give a bias for the left and right crop bbox.
        croped_img = crop(
            img,
            top=y1,
            left=(x1 - bias),
            height=box_height,
            width=(box_width + 2 * bias),
        )

        pad_img = pad(croped_img, padding=(width_gap - bias, 0), fill=0)

        resized_img = resize(pad_img, size=(img_size, img_size))

        object_list.append(resized_img)

    return object_list  # c, t, h, w


def del_folder(path, *args):
    """
    delete the folder which path/version

    Args:
        path (str): path
        version (str): version
    """
    if os.path.exists(os.path.join(path, *args)):
        shutil.rmtree(os.path.join(path, *args))


def make_folder(path, *args):
    """
    make folder which path/version

    Args:
        path (str): path
        version (str): version
    """
    if not os.path.exists(os.path.join(path, *args)):
        os.makedirs(os.path.join(path, *args))
        print("success make dir! where: %s " % os.path.join(path, *args))
    else:
        print("The target path already exists! where: %s " % os.path.join(path, *args))


def merge_frame_to_video(
    save_path: Path, person: str, video_name: str, flag: str, filter: bool = False
) -> None:
    if filter:
        _save_path = save_path / "vis" / "filter_img" / flag / person / video_name
        _out_path = save_path / "vis" / "filter_video" / flag / person
    else:
        _save_path = save_path / "vis" / "img" / flag / person / video_name
        _out_path = save_path / "vis" / "video" / flag / person

    frames = sorted(list(_save_path.iterdir()), key=lambda x: int(x.stem.split("_")[0]))

    if not _out_path.exists():
        _out_path.mkdir(parents=True, exist_ok=True)

    first_frame = cv2.imread(str(frames[0]))
    height, width, _ = first_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        str(_out_path / video_name) + ".mp4", fourcc, 30.0, (width, height)
    )

    for f in tqdm(frames, desc=f"Save {flag}-{video_name}", total=len(frames)):
        img = cv2.imread(str(f))
        out.write(img)

    out.release()

    logger.info(f"Video saved to {_out_path / video_name}.mp4")


def save_to_pt(one_video: Path, save_path: Path, pt_info: dict[torch.Tensor]) -> None:
    """save the sample info to json file.

    Args:
        sample_info (dict): _description_
        save_path (Path): _description_
        logger (logging): _description_
    """

    # change the one_video to video name
    # TODO: change the disease
    if "LCS" in one_video.stem or "HipOA" in one_video.stem:
        one_video = Path(str(one_video).replace("ASD_not", "LCS_HipOA"))
    elif "DHS" in one_video.stem:
        one_video = Path(str(one_video).replace("ASD_not", "DHS"))

    disease = one_video.parts[-2]
    video_name = one_video.stem

    save_path_with_name = save_path / "pt" / disease / (video_name + ".pt")

    make_folder(save_path_with_name.parent)

    torch.save(pt_info, save_path_with_name)

    logger.info(f"Save the {video_name} to {save_path_with_name}")


def save_to_pt_gz(
    one_video: Path, save_path: Path, pt_info: dict[torch.Tensor]
) -> None:
    """Save the sample info to compressed .pt.gz file.

    Args:
        one_video (Path): path to the original video
        save_path (Path): root path to save the compressed .pt.gz
        pt_info (dict): dictionary of tensors or data to be saved
        logger (logging.Logger): logger instance for info output
    """

    # Adjust disease label based on video name
    if "LCS" in one_video.stem or "HipOA" in one_video.stem:
        one_video = Path(str(one_video).replace("ASD_not", "LCS_HipOA"))
    elif "DHS" in one_video.stem:
        one_video = Path(str(one_video).replace("ASD_not", "DHS"))

    disease = one_video.parts[-2]
    video_name = one_video.stem

    save_path_with_name = save_path / "pt_gz" / disease / (video_name + ".pt.gz")
    make_folder(save_path_with_name.parent)

    # 类型转换 + 压缩准备
    for k, v in pt_info.items():
        # print(f"处理 {k}: {type(v)}")

        # 不需处理的基本类型
        if isinstance(v, (str, int, tuple)):
            continue

        # Path 类型转为字符串
        elif isinstance(v, Path):
            pt_info[k] = str(v)

        # Tensor 类型处理
        elif isinstance(v, torch.Tensor):
            if k == "mask":
                pt_info[k] = v.to(torch.uint8)
            elif v.is_floating_point() and v.dtype == torch.float32:
                try:
                    pt_info[k] = v.to(torch.float16)
                except Exception as e:
                    print(f"警告: 无法将 {k} 转换为 float16: {e}")

    # Save with gzip compression
    with gzip.open(save_path_with_name, "wb") as f:
        torch.save(pt_info, f)

    logger.info(f"Compressed save: {video_name} → {save_path_with_name}")


# def process_none(batch_Dict: dict[torch.Tensor], none_index: list):
#     """
#     process_none, where from batch_Dict to instead the None value with next frame tensor (or froward frame tensor).

#     Args:
#         batch_Dict (dict): batch in Dict, where include the None value when yolo dont work.
#         none_index (list): none index list map to batch_Dict, here not use this.

#     Returns:
#         list: list include the replace value for None value.
#     """

#     boundary = len(batch_Dict) - 1
#     filter_batch = batch_Dict.copy()

#     for i in none_index:

#         # * if the index is None, we need to replace it with next frame.
#         if batch_Dict[i] is None:

#             next_idx = i
#             while True:
#                 # * if the next index is None, we need to find the next not None index.
#                 if next_idx < boundary and batch_Dict[next_idx] is None:
#                     next_idx += 1
#                 else:
#                     break

#             if next_idx < boundary:
#                 filter_batch[i] = batch_Dict[next_idx]
#             else:
#                 filter_batch[i] = batch_Dict[boundary-1]

#     return filter_batch


def process_none(
    batch_Dict: Dict[int, torch.Tensor], none_index: List[int]
) -> Dict[int, torch.Tensor]:
    """
    Replace None entries in the batch dictionary with the next available valid tensor.
    If no valid tensor is found ahead, use the last valid tensor before the boundary.

    Args:
        batch_dict (Dict[int, torch.Tensor]): Dictionary of index -> tensor, some entries are None.
        none_index (List[int]): List of indices in batch_dict that are None.

    Returns:
        Dict[int, torch.Tensor]: A new dictionary with None values replaced.
    """
    filtered_batch = copy.deepcopy(batch_Dict)
    max_index = max(batch_Dict.keys())

    for idx in none_index:
        if filtered_batch.get(idx) is not None:
            continue

        # Search forward
        next_idx = idx + 1
        while next_idx <= max_index and filtered_batch.get(next_idx) is None:
            next_idx += 1

        if next_idx <= max_index and filtered_batch.get(next_idx) is not None:
            filtered_batch[idx] = filtered_batch[next_idx]
        else:
            # If no valid frame ahead, try to use the last one
            prev_idx = idx - 1
            while prev_idx >= 0 and filtered_batch.get(prev_idx) is None:
                prev_idx -= 1
            if prev_idx >= 0:
                filtered_batch[idx] = filtered_batch[prev_idx]
            else:
                raise ValueError(f"Cannot find valid replacement for index {idx}")

    return filtered_batch


def process_none_old(batch: torch.tensor, batch_Dict: dict, none_index: list):
    """
    process_none, where from batch_Dict to instead the None value with next frame tensor (or froward frame tensor).

    Args:
        batch_Dict (dict): batch in Dict, where include the None value when yolo dont work.
        none_index (list): none index list map to batch_Dict, here not use this.

    Returns:
        list: list include the replace value for None value.
    """

    boundary = len(batch_Dict) - 1  # 8
    filter_batch = batch

    for k, v in batch_Dict.items():
        if v == None:
            if (
                None in list(batch_Dict.values())[k:]
                and len(set(list(batch_Dict.values())[k:])) == 1
            ):
                next_idx = k - 1
            else:
                next_idx = k + 1
                while batch_Dict[next_idx] == None and next_idx < boundary:
                    next_idx += 1

            batch_Dict[k] = batch_Dict[next_idx]

            # * copy the next frame to none index
            filter_batch[:, :, k, ...] = batch[:, :, next_idx, ...]

    return list(batch_Dict.values()), filter_batch


def timing(name=None, logger=None, level=logging.INFO):
    """
    用于函数的装饰器形式计时器，支持日志输出。
    用法: @timing("函数名", logger)
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            label = name or "_".join(args[2])
            _logger = logger or logging.getLogger(func.__module__)
            start_time = time.time()

            _logger.log(level, f"⏱️ Start: {label}")

            result = func(*args, **kwargs)

            elapsed = time.time() - start_time
            _logger.log(level, f"✅ End: {label} in {elapsed:.3f} sec")
            return result

        return wrapper

    return decorator
