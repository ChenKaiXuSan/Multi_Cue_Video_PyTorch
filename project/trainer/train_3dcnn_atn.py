"""
File: train.py
Project: project
Created Date: 2023-10-19 02:29:47
Author: chenkaixu
-----
Comment:
Thie file is the train/val/test process for 3dcnn with attention branch network (ATN) for the project.

Have a good code time!
-----
Last Modified: Friday April 25th 2025 6:25:57 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
HISTORY:
Date 	By 	Comments
------------------------------------------------

02-05-2025	Kaixu Chen	add raw_attn_map and gen_attn_map.

22-03-2024	Kaixu Chen	add different class number mapping, now the class number is a hyperparameter.

14-12-2023	Kaixu Chen refactor the code, now it a simple code to train video frame from dataloader.

"""

from typing import Any, List, Optional, Union
import os
import logging

import torch
import torch.nn.functional as F
from torchvision.utils import save_image


from pytorch_lightning import LightningModule

from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
    MulticlassConfusionMatrix,
)

from project.models.res_3dcnn_atn import Res3DCNNATN
from project.helper import save_helper

logger = logging.getLogger(__name__)


class Res3DCNNATNTrainer(LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.img_size = hparams.data.img_size
        self.lr = hparams.optimizer.lr

        self.num_classes = hparams.model.model_class_num

        # define 3dcnn with ATN
        self.resnet_atn = Res3DCNNATN(hparams)

        self.save_hyperparameters()

        self._accuracy = MulticlassAccuracy(num_classes=self.num_classes)
        self._precision = MulticlassPrecision(num_classes=self.num_classes)
        self._recall = MulticlassRecall(num_classes=self.num_classes)
        self._f1_score = MulticlassF1Score(num_classes=self.num_classes)
        self._confusion_matrix = MulticlassConfusionMatrix(num_classes=self.num_classes)

    def forward(self, x):
        return self.resnet_atn(x)

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int):

        # prepare the input and label
        video = batch["video"].detach()  # b, c, t, h, w
        raw_attn_map = batch["attn_map"].detach()  # b, c, t, h, w
        label = batch["label"].detach().float()  # b

        b, c, t, h, w = video.shape

        if b > 20:
            video = video[:20, :, :, :, :]
            raw_attn_map = raw_attn_map[:20, :, :, :, :]
            label = label[:20]

        att_opt, per_opt, gen_att_map, fused_ipt = self.resnet_atn(video, raw_attn_map)

        # check shape
        # if b == 1:
        #     label = label.unsqueeze(0)

        assert label.shape[0] == per_opt.shape[0]

        # compute output
        att_loss = F.cross_entropy(att_opt, label.long())
        per_loss = F.cross_entropy(per_opt, label.long())
        attn_map_loss = F.cross_entropy(gen_att_map, raw_attn_map)
        loss = att_loss + per_loss + attn_map_loss

        self.log("train/loss", loss, on_epoch=True, on_step=True, batch_size=b)

        # log metrics
        video_preds_softmax = torch.softmax(per_opt, dim=1)
        video_acc = self._accuracy(video_preds_softmax, label)
        video_precision = self._precision(video_preds_softmax, label)
        video_recall = self._recall(video_preds_softmax, label)
        video_f1_score = self._f1_score(video_preds_softmax, label)

        self.log_dict(
            {
                "train/video_acc": video_acc,
                "train/video_precision": video_precision,
                "train/video_recall": video_recall,
                "train/video_f1_score": video_f1_score,
            },
            on_epoch=True,
            on_step=True,
            batch_size=b,
        )
        logger.info(f"train loss: {loss.item()}")

        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int):

        # input and model define
        video = batch["video"].detach()  # b, c, t, h, w
        raw_attn_map = batch["attn_map"].detach()  # b, c, t, h, w
        label = batch["label"].detach().float() # b

        b, c, t, h, w = video.shape

        att_opt, per_opt, gen_att_map, fused_ipt = self.resnet_atn(video, raw_attn_map)

        # check shape
        # if b == 1:
        #     label = label.unsqueeze(0)

        assert label.shape[0] == per_opt.shape[0]

        # compute output
        att_loss = F.cross_entropy(att_opt, label.long())
        per_loss = F.cross_entropy(per_opt, label.long())
        attn_map_loss = F.cross_entropy(gen_att_map, raw_attn_map)
        loss = att_loss + per_loss + attn_map_loss

        self.log("val/loss", loss, on_epoch=True, on_step=True, batch_size=b)

        # log metrics
        video_preds_softmax = torch.softmax(per_opt, dim=1)
        video_acc = self._accuracy(video_preds_softmax, label)
        video_precision = self._precision(video_preds_softmax, label)
        video_recall = self._recall(video_preds_softmax, label)
        video_f1_score = self._f1_score(video_preds_softmax, label)

        self.log_dict(
            {
                "val/video_acc": video_acc,
                "val/video_precision": video_precision,
                "val/video_recall": video_recall,
                "val/video_f1_score": video_f1_score,
            },
            on_epoch=True,
            on_step=True,
            batch_size=b,
        )

        logger.info(f"val loss: {loss.item()}")

    def save_images(
        self,
        video: torch.Tensor,
        raw_attn_map: torch.Tensor,
        gen_attn_map: torch.Tensor,
        fuse_video: torch.Tensor,
        batch_idx: int = 0,
    ) -> None:

        save_pth = os.path.join(self.logger.root_dir, "imgs")

        b, c, t, h, w = video.shape

        for i in range(b):

            if i >= 2:
                break

            # img
            _pth = os.path.join(save_pth, "raw_img")
            if not os.path.exists(_pth):
                os.makedirs(_pth)
            save_image(
                video[i, :, 0, :, :],
                os.path.join(_pth, f"batch_{batch_idx}_person_{i}.png"),
                normalize=True,
            )
            # raw attn map
            _pth = os.path.join(save_pth, "raw_attn_map")
            if not os.path.exists(_pth):
                os.makedirs(_pth)
            save_image(
                raw_attn_map[i, :, 0, :, :],
                os.path.join(_pth, f"batch_{batch_idx}_person_{i}.png"),
                normalize=True,
            )
            # gen attn map
            _pth = os.path.join(save_pth, "gen_attn_map")
            if not os.path.exists(_pth):
                os.makedirs(_pth)
            save_image(
                gen_attn_map[i, :, 0, :, :],
                os.path.join(_pth, f"batch_{batch_idx}_person_{i}.png"),
                normalize=True,
            )
            # fuse img
            _pth = os.path.join(save_pth, "fuse_img")
            if not os.path.exists(_pth):
                os.makedirs(_pth)
            save_image(
                fuse_video[i, :, 0, :, :],
                os.path.join(_pth, f"batch_{batch_idx}_person_{i}.png"),
                normalize=True,
            )

    ##############
    # test step
    ##############
    # the order of the hook function is:
    # on_test_start -> test_step -> on_test_batch_end -> on_test_epoch_end -> on_test_end

    def on_test_start(self) -> None:
        """hook function for test start"""
        self.test_outputs: list[torch.Tensor] = []
        self.test_pred_list: list[torch.Tensor] = []
        self.test_label_list: list[torch.Tensor] = []

        logger.info("test start")

    def on_test_end(self) -> None:
        """hook function for test end"""
        logger.info("test end")

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int):

        # input and model define
        video = batch["video"].detach()  # b, c, t, h, w
        raw_attn_map = batch["attn_map"].detach()  # b, c, t, h, w
        label = batch["label"].detach().float()  # b

        b, c, t, h, w = video.shape

        att_opt, per_opt, gen_att_map, fused_ipt = self.resnet_atn(video, raw_attn_map)

        # check shape
        # if b == 1:
        #     label = label.unsqueeze(0)
        # assert label.shape[0] == b

        # compute output
        att_loss = F.cross_entropy(att_opt, label.long())
        per_loss = F.cross_entropy(per_opt, label.long())
        attn_map_loss = F.cross_entropy(gen_att_map, raw_attn_map)
        loss = att_loss + per_loss + attn_map_loss

        self.log("test/loss", loss, on_epoch=True, on_step=True, batch_size=b)

        # log metrics
        video_preds_softmax = torch.softmax(per_opt, dim=1)
        video_acc = self._accuracy(video_preds_softmax, label)
        video_precision = self._precision(video_preds_softmax, label)
        video_recall = self._recall(video_preds_softmax, label)
        video_f1_score = self._f1_score(video_preds_softmax, label)

        metric_dict = {
            "test/video_acc": video_acc,
            "test/video_precision": video_precision,
            "test/video_recall": video_recall,
            "test/video_f1_score": video_f1_score,
        }
        self.log_dict(metric_dict, on_epoch=True, on_step=True, batch_size=b)

        # save imgs
        self.save_images(video, raw_attn_map, gen_att_map, fused_ipt, batch_idx)

        return att_opt, video_preds_softmax, gen_att_map

    def on_test_batch_end(
        self,
        outputs: list[torch.Tensor],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """hook function for test batch end

        Args:
            outputs (torch.Tensor | logging.Mapping[str, Any] | None): current output from batch.
            batch (Any): the data of current batch.
            batch_idx (int): the index of current batch.
            dataloader_idx (int, optional): the index of all dataloader. Defaults to 0.
        """

        att_opt, per_opt, attention = outputs
        label = batch["label"].detach().float()

        self.test_outputs.append(outputs)
        self.test_pred_list.append(per_opt)
        self.test_label_list.append(label)

    def on_test_epoch_end(self) -> None:
        """hook function for test epoch end"""
        # save confusion matrix
        save_helper(
            all_pred=self.test_pred_list,
            all_label=self.test_label_list,
            fold=self.logger.root_dir.split("/")[-1],
            save_path=self.logger.save_dir,
            num_class=self.num_classes,
        )

        logger.info("test epoch end")

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
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=self.trainer.estimated_stepping_batches,
                    verbose=True,
                ),
                "monitor": "train/loss",
            },
        }
