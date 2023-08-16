'''
File: train.py
Project: project
Created Date: 2023-08-15 10:51:51
Author: chenkaixu
-----
Comment:
train process for project.
from main.py, define the train and val process logic.
 
Have a good code time!
-----
Last Modified: 2023-08-16 15:58:40
Modified By: chenkaixu
-----
HISTORY:
Date 	By 	Comments
------------------------------------------------
2023-08-15	KX.C	make file, see TODO for next step.

'''

# %%
import csv, logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pytorch_lightning import LightningModule

from torchvision.io import write_video
from torchvision.utils import save_image, flow_to_image

from torchmetrics import classification
from torchmetrics.functional.classification import \
    binary_f1_score, \
    binary_accuracy, \
    binary_cohen_kappa, \
    binary_auroc, \
    binary_confusion_matrix

from models.make_model import MakeVideoModule
from models.preprocess import Preprocess

# %%
class MultiCueLightningModule(LightningModule):

    def __init__(self, hparams):
        super().__init__()

        self.img_size = hparams.data.img_size
        self.lr = hparams.optimizer.lr
        
        self.preprocess = Preprocess(hparams)
        self.cnn = MakeVideoModule(hparams).make_walk_resnet(3)
        self.num_classes = hparams.model.model_class_num

        # save the hyperparameters to the file and ckpt
        self.save_hyperparameters()

        # TODO change the num classes num 
        # select the metrics
        # self._accuracy = classification.MulticlassAccuracy(num_classes=self.num_classes)
        # self._precision = classification.MulticlassPrecision(num_classes=self.num_classes)
        # self._confusion_matrix = classification.MulticlassConfusionMatrix(num_classes=self.num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch: torch.Tensor, batch_idx:int):
        # input and model define
        video = batch['video'].detach()  # b, c, t, h, w
        label = batch['label'].detach()  # b

        b, c, t, h, w = video.shape

        optical_flow, mask, pose = self.preprocess(video, batch_idx)


    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        '''
        val step when trainer.fit called.

        Args:
            batch (torch.Tensor): b, f, h, w
            batch_idx (int): batch index, or patient index

        Returns: None
        '''

        # input and model define
        video = batch['video'].detach()  # b, c, t, h, w
        label = batch['label'].detach()  # b

        b, c, t, h, w = video.shape

        # ? 做不同融合的时候，需要不同的信息，这个怎么办
        optical_flow, mask, pose = self.preprocess(video, batch_idx)

        # pred the video frames
        # with torch.no_grad():
        #     bat_pred, DVF = self.model(invol, rpm_x=test_x_rpm_tensor, rpm_y=test_y_rpm_tensor, future_seq=self.seq)  # [1,2,3,176,176]

        # # calc loss 
        # phase_mse_loss_list = []
        # phase_smooth_l1_loss_list = []

        # for phase in range(self.seq):
        #     phase_mse_loss_list.append(F.mse_loss(bat_pred[:,:,phase,...], batch.expand_as(bat_pred[:,:,phase,...])))
        #     phase_smooth_l1_loss_list.append(F.smooth_l1_loss(DVF[:,:,phase,...], batch.expand_as(DVF[:,:,phase,...])))
        
        # val_loss = torch.mean(torch.stack(phase_mse_loss_list,dim=0)) + torch.mean(torch.stack(phase_smooth_l1_loss_list, dim=0))

        # self.log('val_loss', val_loss)
        # logging.info('val_loss: %d' % val_loss)

    def configure_optimizers(self):
        '''
        configure the optimizer and lr scheduler

        Returns:
            optimizer: the used optimizer.
            lr_scheduler: the selected lr scheduler.
        '''

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer),
                "monitor": "val_loss",
            },
        }
        # return torch.optim.SGD(self.parameters(), lr=self.lr)

    def _get_name(self):
        return self.model_type
