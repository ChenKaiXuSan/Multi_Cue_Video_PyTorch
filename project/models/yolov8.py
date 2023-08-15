'''
File: yolov8.py
Project: models
Created Date: 2023-08-01 12:45:22
Author: chenkaixu
-----
Comment:
Yolov8 method for objection detection and segmentation, pose estimation.
The yolov8 ckpt from ultralytics. 
 
Have a good code time!
-----
Last Modified: 2023-08-15 15:47:01
Modified By: chenkaixu
-----
HISTORY:
Date 	By 	Comments
------------------------------------------------

'''

# %%
import torch
from torchvision.transforms.functional import crop, pad, resize
from torchvision.io import read_video

import os
import shutil
import sys
import multiprocessing
from argparse import ArgumentParser

import matplotlib.pyplot as plt

from ultralytics import YOLO


class MultiPreprocess(torch.nn.Module):

    def __init__(self, configs) -> None:
        super().__init__()

        # load model
        self.yolo_pose = YOLO(configs.pose_ckpt)
        self.yolo_mask = YOLO(configs.seg_ckpt)

        self.conf = configs.conf
        self.iou = configs.iou

    def get_YOLO_pose_result(self, frame_batch):

        c, t, h, w = frame_batch.shape

        with torch.no_grad():
            results = self.yolo_pose(source=frame_batch.permute(
                1, 0, 2, 3), conf=self.conf, iou=self.iou, save_crop=False, classes=0, vid_stride=True, stream=False)

        one_batch_keypoint = []

        for i, r in enumerate(results):
            # one_batch_keypoint.append(r.keypoints.data) # 1, 17, 3
            one_batch_keypoint.append(r.keypoints.xy)  # 1, 17

        return one_batch_keypoint

    def get_YOLO_mask_result(self, frame_batch):

        c, t, h, w = frame_batch.shape

        with torch.no_grad():
            results = self.yolo_mask(source=frame_batch.permute(
                1, 0, 2, 3), conf=self.conf, iou=self.iou, save_crop=False, classes=0, vid_stride=True, stream=False)

        one_batch_mask = []

        # TODO need to treat when r is None. (no results.)
        for i, r in enumerate(results):
            one_batch_mask.append(r.masks.data)  # 1, 224, 224

        return one_batch_mask

    def process_batch(self, batch: torch.Tensor):

        b, c, t, h, w = batch.shape

        pred_mask_list = []
        pred_keypoint_list = []

        for batch_index in range(b):
            one_batch_mask = self.get_YOLO_mask_result(batch[batch_index])
            one_batch_keypoint = self.get_YOLO_pose_result(batch[batch_index])

            pred_mask_list.append(torch.stack(
                one_batch_mask, dim=1))  # c, t, h, w
            pred_keypoint_list.append(torch.stack(
                one_batch_keypoint, dim=0))  # b, c, keypoint, value

        # return mask, keypoint
        return torch.stack(pred_mask_list, dim=0), torch.stack(pred_keypoint_list, dim=0)

    def forward(self, batch):

        return self.process_batch(batch)
