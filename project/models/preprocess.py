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
Last Modified: 2023-08-16 16:43:10
Modified By: chenkaixu
-----
HISTORY:
Date 	By 	Comments
------------------------------------------------
2023-08-15	KX.C	some promble with forward function.

'''

import shutil
from pathlib import Path
from typing import Any
import torch  
import torch.nn as nn 
import torch.nn.functional as F 
from torchvision.io import write_png
from torchvision.utils import flow_to_image

from optical_flow import OpticalFlow
from yolov8 import  MultiPreprocess
from make_model import MakeVideoModule

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
        # only rmtree in 1 epoch, and in one flag (optical flow).
        if pref_Path.exists() and epoch_idx == 0 and flag == 'optical_flow':
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
                
        print('finish save %s' % flag)

    def forward(self, batch: torch.tensor, batch_idx: int):
        """
        forward preprocess method for one batch, use yolo and RAFT.

        Args:
            batch (torch.tensor): batch imgs, (b, c, t, h, w)
            batch_idx (int): epoch index.

        Returns:
            list: list for different moddailty, return optical flow, mask, and pose keypoints.
        """
                
        b, c, t, h, w = batch.shape

        # ? 这两种方法预测出来的图片，好像序列对不上（batch对不上）
        # * slove, 因为不同epoch覆盖了，加一个batch_idx在最开始防止覆盖
        # ? raft这个方法有问题，有些图片预测出来的样子很奇怪。感觉是transform的问题，但是确认了代码，没什么问题。
        
        optical_flow = self.of_model(batch)
        self.save_img(optical_flow, flag='optical_flow', epoch_idx=batch_idx)
        mask, pose = self.yolo_model(batch)
        self.save_img(mask, flag='mask', epoch_idx=batch_idx)
        
        return optical_flow, mask, pose


