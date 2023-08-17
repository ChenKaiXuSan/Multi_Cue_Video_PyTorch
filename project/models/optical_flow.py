'''
File: optical_flow.py
Project: models
Created Date: 2023-08-01 10:17:50
Author: chenkaixu
-----
Comment:
optical flow file, use RAFT to predict the given img patch (first img, second img), get the predicted optical flow.

Have a good code time!
-----
Last Modified: 2023-08-17 14:43:14
Modified By: chenkaixu
-----
HISTORY:
Date 	By 	Comments
------------------------------------------------
2023-08-17	KX.C	add comment for function.

'''

import numpy as np 
import torch
import torch.nn as nn
import torchvision.transforms.functional as F

from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

class OpticalFlow(nn.Module):

    def __init__(self):
        super(OpticalFlow, self).__init__()

        self.weights = Raft_Large_Weights.DEFAULT
        self.transforms = self.weights.transforms()

        #define the network 
        self.optical_flow = raft_large(weights=self.weights, progress=False).cuda()
        
    def get_Optical_flow(self, frame_batch: torch.tensor):
        '''
        catch one by one batch optical flow, use RAFT method.

        Args:
            frame_batch (tensor): one batch frame, (c, f, h, w)

        Returns:
            tensor: one batch pred optical flow 
        '''

        c, f, h, w = frame_batch.shape

        frame_batch = frame_batch.permute(1, 0, 2, 3) # c, f, h, w to f, c, h, w

        current_frame = frame_batch[:-1, :, :, :] # 0~-1 frame
        next_frame = frame_batch[1:, :, :, :] # 1~last frame

        # transforms
        current_frame_batch, next_frame_batch = self.transforms(current_frame, next_frame)

        # start predict 
        self.optical_flow.eval()

        with torch.no_grad():
            pred_flows = self.optical_flow(current_frame_batch, next_frame_batch)[-1]

        # empty cache
        torch.cuda.empty_cache()

        # copy the final optical flow to pred_flows, because need from 7 to 8 
        pred_flows = torch.cat([pred_flows, pred_flows[-1, ...].unsqueeze(dim=0)], dim=0) # f, c, h, w
        return pred_flows # f, c, h, w
    
    def process_batch(self, batch: torch.tensor):
        '''
        predict one batch optical flow.

        Args:
            batch (torch.tensor): batches of videos. (b, c, f, h, w)

        Returns:
            torch.tensor: stacked predict optical flow, (b, 2, f, h, w)
        '''        
        
        b, c, f, h, w = batch.shape

        pred_optical_flow_list = []

        for batch_index in range(b):
            one_batch_pred_flow = self.get_Optical_flow(batch[batch_index]) # f, c, h, w
            pred_optical_flow_list.append(one_batch_pred_flow) 
    
        return torch.stack(pred_optical_flow_list).permute(0, 2, 1, 3, 4) # b, c, f, h, w
    
    def forward(self, batch):

        b, c, t, h, w = batch

        of = self.process_batch(batch)

        # shape check 
        assert of.shape[0] == b
        assert of.shape[2] == t

        return of