"""
File: main.py
Project: project
Created Date: 2023-08-01 10:17:50
Author: chenkaixu
-----
this project were based the pytorch, pytorch lightning and pytorch video library, 
for rapid development.
-----
Last Modified: 2023-08-29 17:03:58
Modified By: chenkaixu
-----
HISTORY:
Date 	By 	Comments
------------------------------------------------
2023-08-15	KX.C	make the file.

"""
# %%
import os, time, warnings, sys, logging, multiprocessing

warnings.filterwarnings("ignore")

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning import loggers as pl_loggers

# callbacks
from pytorch_lightning.callbacks import (
    TQDMProgressBar,
    RichModelSummary,
    ModelCheckpoint,
    EarlyStopping,
)
from pl_bolts.callbacks import PrintTableMetricsCallback, TrainingDataMonitor

from dataloader.data_loader import WalkDataModule
from train import MultiCueLightningModule
from dataloader import WalkDataModule

import pytorch_lightning
import hydra


# %%
def train(hparams, fold: int):
    # set seed
    seed_everything(42, workers=True)

    # instance the dataset
    data_module = WalkDataModule(hparams)

    # instance the model
    classification_module = MultiCueLightningModule(hparams)

    # for the tensorboard
    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=os.path.join(hparams.train.log_path), version=fold
    )

    # some callbacks
    progress_bar = TQDMProgressBar(refresh_rate=100)
    rich_model_summary = RichModelSummary(max_depth=2)

    # define the checkpoint becavier.
    model_check_point = ModelCheckpoint(
        filename="{epoch}-{val_loss:.2f}-{val_video_acc:.4f}-{val_of_acc:.4f}-{val_mask_acc:.4f}",
        auto_insert_metric_name=True,
        monitor="val_loss",
        mode="min",
        save_last=False,
        save_top_k=2,
    )

    # define the early stop.
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=5,
        mode="min",
    )

    # bolts callbacks
    table_metrics_callback = PrintTableMetricsCallback()
    monitor = TrainingDataMonitor(log_every_n_steps=50)

    trainer = Trainer(
        devices=[
            hparams.train.gpu_num,
        ],
        accelerator="gpu",
        max_epochs=hparams.train.max_epochs,
        logger=tb_logger,
        #   log_every_n_steps=100,
        check_val_every_n_epoch=1,
        callbacks=[
            progress_bar,
            rich_model_summary,
            table_metrics_callback,
            monitor,
            model_check_point,
            early_stopping,
        ],
        #   deterministic=True
    )

    # training and val
    trainer.fit(classification_module, data_module)

    Acc_list = trainer.validate(classification_module, data_module, ckpt_path="best")

    # return the best acc score.
    # return model_check_point.best_model_score.item()


@hydra.main(
    version_base=None,
    config_path="/workspace/Multi_Cue_Video_PyTorch/configs",
    config_name="config.yaml",
)
def init_params(
    config,
):
    #############
    # K Fold CV
    #############
    DATE = str(time.localtime().tm_mon) + str(time.localtime().tm_mday)
    DATA_PATH = config.data.data_path

    # set the version
    uniform_temporal_subsample_num = config.train.uniform_temporal_subsample_num
    clip_duration = config.train.clip_duration
    config.train.version = "_".join(
        [DATE, str(clip_duration), str(uniform_temporal_subsample_num)]
    )

    # output log to file
    log_path = (
        "/workspace/Multi_Cue_Video_PyTorch/logs/"
        + "_".join([config.train.version, config.model.model])
        + ".log"
    )
    sys.stdout = open(log_path, "w")

    # get the fold number
    fold_num = os.listdir(DATA_PATH)
    fold_num.sort()
    if "raw" in fold_num:
        fold_num.remove("raw")

    store_Acc_Dict = {}
    sum_list = []

    for fold in fold_num:
        #################
        # start k Fold CV
        #################

        logging.info("#" * 50)
        logging.info("Start %s" % fold)
        logging.info("#" * 50)

        config.train.train_path = os.path.join(DATA_PATH, fold)

        Acc_score = train(config, fold)

        store_Acc_Dict[fold] = Acc_score
        sum_list.append(Acc_score)

    logging.info("#" * 50)
    logging.info("different fold Acc:")
    logging.info(store_Acc_Dict)
    logging.info("Final avg Acc is: %s" % (sum(sum_list) / len(sum_list)))


# %%
if __name__ == "__main__":
    # # define process
    # ASD_process = multiprocessing.Process(target=main, args=(parames, 'ASD'))
    # DHS_process = multiprocessing.Process(target=main, args=(parames, 'DHS'))
    # LCS_process = multiprocessing.Process(target=main, args=(parames, 'LCS'))
    # HipOA_process = multiprocessing.Process(target=main, args=(parames, 'HipOA'))

    # # start process
    # ASD_process.start()
    # DHS_process.start()
    # LCS_process.start()
    # HipOA_process.start()

    os.environ["HYDRA_FULL_ERROR"] = "1"
    init_params()
