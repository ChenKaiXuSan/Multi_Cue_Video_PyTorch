"""
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
Last Modified: 2023-08-29 16:52:24
Modified By: chenkaixu
-----
HISTORY:
Date 	By 	Comments
------------------------------------------------
2023-08-15	KX.C	make file, see TODO for next step.

"""

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
from torchmetrics.functional.classification import (
    binary_f1_score,
    binary_accuracy,
    binary_cohen_kappa,
    binary_auroc,
    binary_confusion_matrix,
)

from models.make_model import MakeVideoModule
from models.preprocess import Preprocess


# %%
class MultiCueLightningModule(LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.img_size = hparams.data.img_size
        self.lr = hparams.optimizer.lr

        self.preprocess = Preprocess(hparams)
        self.num_classes = hparams.model.model_class_num

        # define model
        self.video_cnn = MakeVideoModule(hparams).make_walk_resnet(3)
        self.of_cnn = MakeVideoModule(hparams).make_walk_resnet(2)
        self.mask_cnn = MakeVideoModule(hparams).make_walk_resnet(1)

        # save the hyperparameters to the file and ckpt
        self.save_hyperparameters()

        # TODO change the num classes num
        # select the metrics
        # self._accuracy = classification.MulticlassAccuracy(num_classes=self.num_classes)
        # self._precision = classification.MulticlassPrecision(num_classes=self.num_classes)
        # self._confusion_matrix = classification.MulticlassConfusionMatrix(num_classes=self.num_classes)

        self._accuracy = classification.BinaryAccuracy()
        self._precision = classification.BinaryPrecision()
        self._recall = classification.BinaryRecall()
        self._confusion_matrix = classification.BinaryConfusionMatrix()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        # input and model define
        video = batch["video"].detach()  # b, c, t, h, w
        label = batch["label"].detach()  # b

        b, c, t, h, w = video.shape

        video, label, optical_flow, mask, pose = self.preprocess(
            video, label, batch_idx
        )
        # logging.info('*' * 50)
        # logging.info([video.shape, label.shape, optical_flow.shape, mask.shape, pose.shape])

        # input and model define
        video = batch["video"].detach()  # b, c, t, h, w
        label = batch["label"].detach().float()  # b

        b, c, t, h, w = video.shape

        # ? 做不同融合的时候，需要不同的信息，这个怎么办
        video, label, optical_flow, mask, pose = self.preprocess(
            video, label, batch_idx
        )
        logging.info("*" * 50)
        logging.info(
            [video.shape, label.shape, optical_flow.shape, mask.shape, pose.shape]
        )

        video_preds = self.video_cnn(video).squeeze()
        of_preds = self.of_cnn(optical_flow).squeeze()
        mask_preds = self.mask_cnn(mask).squeeze()

        video_preds_sigmoid = torch.sigmoid(video_preds)
        of_preds_sigmoid = torch.sigmoid(of_preds)
        mask_preds_sigmoid = torch.sigmoid(mask_preds)

        video_loss = F.binary_cross_entropy_with_logits(video_preds, label)
        of_loss = F.binary_cross_entropy_with_logits(of_preds, label)
        mask_loss = F.binary_cross_entropy_with_logits(mask_preds, label)

        total_loss = (video_loss + of_loss + mask_loss) / 3
        self.log("train_loss", total_loss)

        # video metrics
        train_video_acc = self._accuracy(video_preds_sigmoid, label)
        train_video_precision = self._precision(video_preds_sigmoid, label)
        train_video_recall = self._recall(video_preds_sigmoid, label)
        train_video_confusion_matrix = self._confusion_matrix(
            video_preds_sigmoid, label
        )

        self.log_dict(
            {
                "train_video_acc": train_video_acc,
                "train_video_precision": train_video_precision,
                "train_video_recall": train_video_recall,
            }
        )
        logging.info("*" * 50)
        logging.info("train_video_confusion_matrix: %s" % train_video_confusion_matrix)

        # of metrics
        train_of_acc = self._accuracy(of_preds_sigmoid, label)
        train_of_precision = self._precision(of_preds_sigmoid, label)
        train_of_recall = self._recall(of_preds_sigmoid, label)
        train_of_confusion_matrix = self._confusion_matrix(of_preds_sigmoid, label)

        self.log_dict(
            {
                "train_of_acc": train_of_acc,
                "train_of_precision": train_of_precision,
                "train_of_recall": train_of_recall,
            }
        )

        logging.info("*" * 50)
        logging.info("train_of_confusion_matrix: %s" % train_of_confusion_matrix)

        # mask metrics
        train_mask_acc = self._accuracy(mask_preds_sigmoid, label)
        train_mask_precision = self._precision(mask_preds_sigmoid, label)
        train_mask_recall = self._recall(mask_preds_sigmoid, label)
        train_mask_confusion_matrix = self._confusion_matrix(mask_preds_sigmoid, label)

        self.log_dict(
            {
                "train_mask_acc": train_mask_acc,
                "train_mask_precision": train_mask_precision,
                "train_mask_recall": train_mask_recall,
            }
        )

        logging.info("*" * 50)
        logging.info("train_mask_confusion_matrix: %s" % train_mask_confusion_matrix)

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        # input and model define
        video = batch["video"].detach()  # b, c, t, h, w
        label = batch["label"].detach().float()  # b

        b, c, t, h, w = video.shape

        # ? 做不同融合的时候，需要不同的信息，这个怎么办
        video, label, optical_flow, mask, pose = self.preprocess(
            video, label, batch_idx
        )
        # logging.info('*' * 50)
        # logging.info([video.shape, label.shape, optical_flow.shape, mask.shape, pose.shape])

        video_preds = self.video_cnn(video).squeeze()
        of_preds = self.of_cnn(optical_flow).squeeze()
        mask_preds = self.mask_cnn(mask).squeeze()

        video_preds_sigmoid = torch.sigmoid(video_preds)
        of_preds_sigmoid = torch.sigmoid(of_preds)
        mask_preds_sigmoid = torch.sigmoid(mask_preds)

        video_loss = F.binary_cross_entropy_with_logits(video_preds, label)
        of_loss = F.binary_cross_entropy_with_logits(of_preds, label)
        mask_loss = F.binary_cross_entropy_with_logits(mask_preds, label)

        total_loss = (video_loss + of_loss + mask_loss) / 3
        self.log("val_loss", total_loss)

        # video metrics
        val_video_acc = self._accuracy(video_preds_sigmoid, label)
        val_video_precision = self._precision(video_preds_sigmoid, label)
        val_video_recall = self._recall(video_preds_sigmoid, label)
        val_video_confusion_matrix = self._confusion_matrix(video_preds_sigmoid, label)

        self.log_dict(
            {
                "val_video_acc": val_video_acc,
                "val_video_precision": val_video_precision,
                "val_video_recall": val_video_recall,
            }
        )

        logging.info("*" * 50)
        logging.info("val_video_confusion_matrix: %s" % val_video_confusion_matrix)

        # of metrics
        val_of_acc = self._accuracy(of_preds_sigmoid, label)
        val_of_precision = self._precision(of_preds_sigmoid, label)
        val_of_recall = self._recall(of_preds_sigmoid, label)
        val_of_confusion_matrix = self._confusion_matrix(of_preds_sigmoid, label)

        self.log_dict(
            {
                "val_of_acc": val_of_acc,
                "val_of_precision": val_of_precision,
                "val_of_recall": val_of_recall,
            }
        )

        logging.info("*" * 50)
        logging.info("val_of_confusion_matrix: %s" % val_of_confusion_matrix)

        # mask metrics
        val_mask_acc = self._accuracy(mask_preds_sigmoid, label)
        val_mask_precision = self._precision(mask_preds_sigmoid, label)
        val_mask_recall = self._recall(mask_preds_sigmoid, label)
        val_mask_confusion_matrix = self._confusion_matrix(mask_preds_sigmoid, label)

        self.log_dict(
            {
                "val_mask_acc": val_mask_acc,
                "val_mask_precision": val_mask_precision,
                "val_mask_recall": val_mask_recall,
            }
        )

        logging.info("*" * 50)
        logging.info("val_mask_confusion_matrix: %s" % val_mask_confusion_matrix)

    def configure_optimizers(self):
        """
        configure the optimizer and lr scheduler

        Returns:
            optimizer: the used optimizer.
            lr_scheduler: the selected lr scheduler.
        """

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        return {
            "optimizer": optimizer,
            # "lr_scheduler": {
            #     "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer),
            #     "monitor": "val_loss",
            # },
        }
        # return torch.optim.SGD(self.parameters(), lr=self.lr)

    def _get_name(self):
        return self.model_type
