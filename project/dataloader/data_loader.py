"""
File: data_loader.py
Project: dataloader
Created Date: 2023-10-19 02:24:47
Author: chenkaixu
-----
Comment:
A pytorch lightning data module based dataloader, for train/val/test dataset prepare.
Have a good code time!
-----
Last Modified: 2023-10-29 12:18:59
Modified By: chenkaixu
-----
HISTORY:
Date 	By 	Comments
------------------------------------------------

12-05-2024	Kaixu Chen	Now choose different kinds of experiments based on the EXPERIMENT keyword.
temporal mix, late fusion and single (stance/swing/random)

04-04-2024	Kaixu Chen	when use temporal mix, need keep same data process method for train/val dataset.

25-03-2024	Kaixu Chen	change batch size for train/val dataloader. Now set bs=1 for gait cycle dataset, set bs=32 for default dataset (without gait cycle).
Because in experiment, I found bs=1 will have large loss when train. Maybe also need gait cycle datset in val/test dataloader?

22-03-2024	Kaixu Chen	add different class num mapping dict. In collate_fn, re-mapping the label from .json disease key.

"""

from torchvision.transforms import (
    Compose,
    Resize,
)

from typing import Any, Callable, Dict, Optional
from pytorch_lightning import LightningDataModule

import torch
from torch.utils.data import DataLoader

from pytorchvideo.data import make_clip_sampler
from pytorchvideo.data.labeled_video_dataset import labeled_video_dataset

from project.dataloader.whole_video_dataset import whole_video_dataset

from project.dataloader.utils import (
    Div255,
    UniformTemporalSubsample,
    ApplyTransformToKey,
)

disease_to_num_mapping_Dict: Dict = {
    2: {"ASD": 0, "non-ASD": 1},
    3: {"ASD": 0, "DHS": 1, "LCS_HipOA": 2},
    4: {"ASD": 0, "DHS": 1, "LCS_HipOA": 2, "normal": 3},
}


class WalkDataModule(LightningDataModule):
    def __init__(self, opt, dataset_idx: Dict):
        super().__init__()

        self._batch_size = opt.data.batch_size

        self._num_workers = opt.data.num_workers
        self._img_size = opt.data.img_size

        # frame rate
        self._clip_duration = opt.train.clip_duration
        self.uniform_temporal_subsample_num = opt.train.uniform_temporal_subsample_num

        # * this is the dataset idx, which include the train/val dataset idx.
        self._dataset_idx = dataset_idx

        self._class_num = opt.model.model_class_num

        self._experiment = opt.train.experiment
        self._backbone = opt.model.backbone
        self._modal_type = opt.train.modal_type

        self.mapping_transform = Compose(
            [
                UniformTemporalSubsample(self.uniform_temporal_subsample_num),
                Div255(),
                Resize(size=[self._img_size, self._img_size]),
            ]
        )

        self.train_video_transform = Compose(
            [
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            Div255(),
                            Resize(size=[self._img_size, self._img_size]),
                            UniformTemporalSubsample(
                                self.uniform_temporal_subsample_num
                            ),
                        ]
                    ),
                ),
            ]
        )

        self.val_video_transform = Compose(
            [
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            Div255(),
                            Resize(size=[self._img_size, self._img_size]),
                            UniformTemporalSubsample(
                                self.uniform_temporal_subsample_num
                            ),
                        ]
                    ),
                ),
            ]
        )

    def prepare_data(self) -> None:
        """here prepare the temp val data path,
        because the val dataset not use the gait cycle index,
        so we directly use the pytorchvideo API to load the video.
        AKA, use whole video to validate the model.
        """
        ...

    def setup(self, stage: Optional[str] = None) -> None:
        """
        assign tran, val, predict datasets for use in dataloaders

        Args:
            stage (Optional[str], optional): trainer.stage, in ('fit', 'validate', 'test', 'predict'). Defaults to None.
        """

        if self._modal_type == "all":
            # train dataset
            self.train_gait_dataset = whole_video_dataset(
                experiment=self._experiment,
                dataset_idx=self._dataset_idx[
                    0
                ],  # train mapped path, include gait cycle index.
                transform=self.mapping_transform,
            )

            # val dataset
            self.val_gait_dataset = whole_video_dataset(
                experiment=self._experiment,
                dataset_idx=self._dataset_idx[
                    1
                ],  # val mapped path, include gait cycle index.
                transform=self.mapping_transform,
            )

            # test dataset
            self.test_gait_dataset = whole_video_dataset(
                experiment=self._experiment,
                dataset_idx=self._dataset_idx[
                    1
                ],  # val mapped path, include gait cycle index.
                transform=self.mapping_transform,
            )

        # TODO: want to delete this part, but need to keep it for now.
        # elif :
        #     # train dataset
        #     self.train_gait_dataset = labeled_video_dataset(
        #         data_path=self._dataset_idx[2],
        #         clip_sampler=make_clip_sampler("uniform", self._clip_duration),
        #         transform=self.train_video_transform,
        #     )

        #     # val dataset
        #     self.val_gait_dataset = labeled_video_dataset(
        #         data_path=self._dataset_idx[3],
        #         clip_sampler=make_clip_sampler("uniform", self._clip_duration),
        #         transform=self.val_video_transform,
        #     )

        #     # test dataset
        #     self.test_gait_dataset = labeled_video_dataset(
        #         data_path=self._dataset_idx[3],
        #         clip_sampler=make_clip_sampler("uniform", self._clip_duration),
        #         transform=self.val_video_transform,
        #     )

    def collate_fn(self, batch):
        """this function process the batch data, and return the batch data.

        Args:
            batch (list): the batch from the dataset.
            The batch include the one patient info from the json file.
            Here we only cat the one patient video tensor, and label tensor.

        Returns:
            dict: {video: torch.tensor, label: torch.tensor, info: list}
        """

        batch_label = []
        batch_rgb = []
        batch_flow = []
        batch_kpt_heatmap = []
        batch_mask = []

        # * mapping label
        for i in batch:

            B, C, T, H, W = i["rgb"].shape

            batch_rgb.append(i["rgb"])  # b, c, t, h, w
            batch_flow.append(i["flow"])  # b, c, t, h, w
            batch_kpt_heatmap.append(i["kpt_heatmap"])  # b, c, t, h, w
            batch_mask.append(i["mask"])  # b, c

            for _ in range(B):

                batch_label.append(
                    i['label']
                )
                
        return {
            "label": torch.tensor(batch_label), # b,
            "rgb": torch.cat(batch_rgb, dim=0), # b, c, t, h, w
            "flow": torch.cat(batch_flow, dim=0),  # b, c, t, h, w
            "kpt_heatmap": torch.cat(batch_kpt_heatmap, dim=0),  # b, c, t, h, w
            "mask": torch.cat(batch_mask, dim=0),  # b, c, t, h, w
        }

    def train_dataloader(self) -> DataLoader:
        """
        create the Walk train partition from the list of video labels
        in directory and subdirectory. Add transform that subsamples and
        normalizes the video before applying the scale, crop and flip augmentations.
        """

        train_data_loader = DataLoader(
            self.train_gait_dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            pin_memory=True,
            shuffle=True,
            drop_last=True,
            collate_fn=self.collate_fn,
        )

        return train_data_loader

    def val_dataloader(self) -> DataLoader:
        """
        create the Walk train partition from the list of video labels
        in directory and subdirectory. Add transform that subsamples and
        normalizes the video before applying the scale, crop and flip augmentations.
        """

        val_data_loader = DataLoader(
            self.val_gait_dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            pin_memory=True,
            shuffle=False,
            drop_last=True,
            collate_fn=self.collate_fn,
        )

        return val_data_loader

    def test_dataloader(self) -> DataLoader:
        """
        create the Walk train partition from the list of video labels
        in directory and subdirectory. Add transform that subsamples and
        normalizes the video before applying the scale, crop and flip augmentations.
        """

        test_data_loader = DataLoader(
            self.test_gait_dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            pin_memory=True,
            shuffle=False,
            drop_last=True,
            collate_fn=self.collate_fn,
        )

        return test_data_loader
