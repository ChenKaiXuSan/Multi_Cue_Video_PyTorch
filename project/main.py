#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/project/main.py
Project: /workspace/code/project
Created Date: Tuesday April 22nd 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Thursday May 1st 2025 8:34:05 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

import os
import logging
import hydra
from omegaconf import DictConfig

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import (
    TQDMProgressBar,
    RichModelSummary,
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)

from project.dataloader.data_loader import WalkDataModule

#####################################
# select different experiment trainer
#####################################

from project.trainer.train_3dcnn import MultiModal3DCNNTrainer
from project.trainer.train_vit import MultiModalVitTrainer

from project.cross_validation import DefineCrossValidation

logger = logging.getLogger(__name__)


def train(hparams: DictConfig, dataset_idx, fold: int):
    """the train process for the one fold.

    Args:
        hparams (hydra): the hyperparameters.
        dataset_idx (int): the dataset index for the one fold.
        fold (int): the fold index.

    Returns:
        list: best trained model, data loader
    """

    seed_everything(42, workers=True)

    # * select experiment
    if hparams.model.backbone == "vit":
        classification_module = MultiModalVitTrainer(hparams)
    elif hparams.model.backbone == "3dcnn":
        classification_module = MultiModal3DCNNTrainer(hparams)
    else:
        raise ValueError("the experiment backbone is not supported.")

    data_module = WalkDataModule(hparams, dataset_idx)

    # for the tensorboard
    tb_logger = TensorBoardLogger(
        save_dir=os.path.join(hparams.train.log_path),
        name=str(fold),  # here should be str type.
    )

    # some callbacks
    progress_bar = TQDMProgressBar(refresh_rate=100)
    rich_model_summary = RichModelSummary(max_depth=2)

    # define the checkpoint becavier.
    model_check_point = ModelCheckpoint(
        filename="{epoch}-{val/loss:.2f}-{val/video_acc:.4f}",
        auto_insert_metric_name=False,
        monitor="val/video_acc",
        mode="max",
        save_last=False,
        save_top_k=2,
    )

    # define the early stop.
    early_stopping = EarlyStopping(
        monitor="val/video_acc",
        patience=5,
        mode="max",
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = Trainer(
        devices=[
            int(hparams.train.gpu),
        ],
        accelerator="gpu",
        max_epochs=hparams.train.max_epochs,
        logger=tb_logger,  # wandb_logger,
        check_val_every_n_epoch=1,
        callbacks=[
            progress_bar,
            rich_model_summary,
            model_check_point,
            early_stopping,
            lr_monitor,
        ],
        # fast_dev_run=hparams.train.fast_dev_run,  # if use fast dev run for debug.
        # limit_train_batches=2,
        # limit_val_batches=2,
        # limit_test_batches=2,
    )

    trainer.fit(classification_module, data_module)

    # save the metrics to file
    trainer.test(
        classification_module,
        data_module,
        ckpt_path="best",
    )


@hydra.main(
    version_base=None,
    config_path="../configs",  # * the config_path is relative to location of the python script
    config_name="config.yaml",
)
def init_params(config):
    #######################
    # prepare dataset index
    #######################

    fold_dataset_idx = DefineCrossValidation(config)()

    logger.info("#" * 50)
    logger.info("Start train all fold")
    logger.info("#" * 50)

    #########
    # K fold
    #########
    # * for one fold, we first train/val model, then save the best ckpt preds/label into .pt file.

    for fold, dataset_value in fold_dataset_idx.items():
        logger.info("#" * 50)
        logger.info(f"Start train fold: {fold}")
        logger.info("#" * 50)

        train(config, dataset_value, fold)

        logger.info("#" * 50)
        logger.info(f"finish train fold: {fold}")
        logger.info("#" * 50)

    logging.info("#" * 50)
    logging.info("finish train all fold")
    logging.info("#" * 50)


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    init_params()
