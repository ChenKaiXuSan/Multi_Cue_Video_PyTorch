'''
File: preprocess.py
Project: models
Created Date: 2023-08-15 11:57:30
Author: chenkaixu
-----
Comment:
The python file to process the RGB frames, clac the OF and mask, and return them.
 
Have a good code time!
-----
Last Modified: 2023-08-30 04:48:05
Modified By: chenkaixu
-----
HISTORY:
Date 	By 	Comments
------------------------------------------------
2023-08-17	KX.C	in batch size = 32, one fold will take about 30 min to take the preprocess method (yolo + optical flow).
2023-08-17	KX.C	finish the preprocess method, in forward function.
2023-08-15	KX.C	some promble with forward function.

'''

import shutil, logging
from pathlib import Path
from typing import Any

import torch  
import torch.nn as nn 
import torch.nn.functional as F 
from torchvision.io import write_png
from torchvision.utils import flow_to_image

from optical_flow import OpticalFlow
from yolov8 import  MultiPreprocess

class Preprocess(nn.Module):

    def __init__(self, config) -> None:
        super(Preprocess, self).__init__()
        self.of_model = OpticalFlow()
        self.yolo_model = MultiPreprocess(config.YOLO)
    
    def save_img(self, batch:torch.tensor, flag:str, epoch_idx:int):
        """
        save_img save batch to png, for analysis.

        Args:
            batch (torch.tensor): the batch of imgs (b, c, t, h, w)
            flag (str): flag for filename, ['optical_flow', 'mask']
            batch_idx (int): epoch index.
        """        
        
        pref_Path = Path('/workspace/Multi_Cue_Video_PyTorch/logs/img_test')
        # only rmtree in 1 epoch, and in one flag (mask).
        if pref_Path.exists() and epoch_idx == 0 and flag == 'mask':
            shutil.rmtree(pref_Path)
            pref_Path.mkdir(parents=True)
            
        b, c, t, h, w = batch.shape

        for b_idx in range(b):
            for frame in range(t):
                
                if flag == 'optical_flow':
                    inp = flow_to_image(batch[b_idx, :, frame, ...])
                else:
                    inp = batch[b_idx, :, frame, ...] * 255
                    inp = inp.to(torch.uint8)
                    
                write_png(input=inp.cpu(), filename= str(pref_Path.joinpath(f'{flag}_{epoch_idx}_{b_idx}_{frame}.png')))
        
        logging.info('='*20)
        logging.info('finish save %s' % flag)

    def shape_check(self, check: list):
        """
        shape_check check give value shape

        Args:
            check (list): checked value, in list.
        """        

        # first value in list is video, use this as reference.
        b, c, t, h, w = check[0].shape

        # frame check, we just need start from 1.
        for ck in check[0:]:
            
            # for label shape
            if len(ck.shape) == 1: assert ck.shape[0] == b 
            # for other shape
            else: assert ck.shape[2] == t

    def forward(self, batch: torch.tensor, labels: list, batch_idx: int):
        """
        forward preprocess method for one batch, use yolo and RAFT.

        Args:
            batch (torch.tensor): batch imgs, (b, c, t, h, w)
            labels (torch.tensor): batch labels, (b) # not use.
            batch_idx (int): epoch index.

        Returns:
            list: list for different moddailty, return optical flow, mask, and pose keypoints.
        """
                
        b, c, t, h, w = batch.shape

        # process mask, pose
        video, labels, mask, pose = self.yolo_model(batch, labels)

        # shape check
        self.shape_check([video, labels, mask, pose])

        self.save_img(mask, flag='mask', epoch_idx=batch_idx)

        # process optical flow
        optical_flow = self.of_model(video)
        
        # shape check
        self.shape_check([video, optical_flow])

        self.save_img(optical_flow, flag='optical_flow', epoch_idx=batch_idx)
        
        return video, labels, optical_flow, mask, pose
