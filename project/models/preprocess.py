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
Last Modified: 2023-08-15 16:14:54
Modified By: chenkaixu
-----
HISTORY:
Date 	By 	Comments
------------------------------------------------
2023-08-15	KX.C	some promble with forward function.

'''

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
    
    def save_img(self, batch:torch.tensor, flag:str):

        b, c, t, h, w = batch.shape

        for batch_idx in range(b):
            for frame in range(t):
                
                if flag == 'optical_flow':
                    inp = flow_to_image(batch[batch_idx, :, frame, ...])
                else:
                    inp = batch[batch_idx, :, frame, ...] * 255
                    inp = inp.to(torch.uint8)
                    
                write_png(input=inp.cpu(), filename='/workspace/Multi_Cue_Video_PyTorch/logs/img_test/%s_%s_%s.png' % (flag, batch_idx, frame))
        print('finish save %s' % flag)

    def forward(self, batch: torch.tensor):
        b, c, t, h, w = batch.shape

        # ? 这两种方法预测出来的图片，好像序列对不上（batch对不上）
        # ? raft这个方法有问题，有些图片预测出来的样子很奇怪。感觉是transform的问题，但是确认了代码，没什么问题。
        
        optical_flow = self.of_model(batch)
        self.save_img(optical_flow, flag='optical_flow')
        mask, pose = self.yolo_model(batch)
        self.save_img(mask, flag='mask')
        
        return optical_flow, mask, pose


