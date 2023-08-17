'''
File: yolov8.py
Project: models
Created Date: 2023-08-01 12:45:22
Author: chenkaixu
-----
Comment:
Yolov8 method for objection detection and segmentation, pose estimation.
The yolov8 ckpt from ultralytics. 
 
Have a good code time!
-----
Last Modified: 2023-08-17 14:32:37
Modified By: chenkaixu
-----
HISTORY:
Date 	By 	Comments
------------------------------------------------
2023-08-17	KX.C	finish the logic for different process. Notice here we seperate treat different method None value.
2023-08-17	KX.C	需要分开处理检测None的问题，不能一起处理。

'''

# %%
import torch
import logging

from ultralytics import YOLO


class MultiPreprocess(torch.nn.Module):

    def __init__(self, configs) -> None:
        super().__init__()

        # load model
        self.yolo_pose = YOLO(configs.pose_ckpt)
        self.yolo_mask = YOLO(configs.seg_ckpt)

        self.conf = configs.conf
        self.iou = configs.iou

    def get_YOLO_pose_result(self, frame_batch: torch.tensor):
        """
        get_YOLO_pose_result, from frame_batch, which (c, t, h, w)

        Args:
            frame_batch (torch.tensor): for processed frame batch, (c, t, h, w)

        Returns:
            dict, list: two return value, with one batch keypoint in Dict, and none index in list.
        """

        c, t, h, w = frame_batch.shape

        with torch.no_grad():
            results = self.yolo_pose(source=frame_batch.permute(
                1, 0, 2, 3), conf=self.conf, iou=self.iou, save_crop=False, classes=0, vid_stride=True, stream=False)

        one_batch_keypoint = {}
        none_index = []

        for i, r in enumerate(results):
            # judge if have keypoints.
            # one_batch_keypoint.append(r.keypoints.data) # 1, 17, 3
            if list(r.keypoints.xy.shape) != [1, 17, 2]:
                none_index.append(i)
                one_batch_keypoint[i] = None
            else:
                one_batch_keypoint[i] = r.keypoints.xy  # 1, 17

        return one_batch_keypoint, none_index

    def get_YOLO_mask_result(self, frame_batch: torch.tensor):
        """
        get_YOLO_mask_result, from frame_batch, for mask.

        Args:
            frame_batch (torch.tensor): for processed frame batch, (c, t, h, w)

        Returns:
            dict, list: two return values, with one batch mask in Dict, and none index in list.
        """

        c, t, h, w = frame_batch.shape

        with torch.no_grad():
            results = self.yolo_mask(source=frame_batch.permute(
                1, 0, 2, 3), conf=self.conf, iou=self.iou, save_crop=False, classes=0, vid_stride=True, stream=False)

        one_batch_mask = {}
        none_index = []

        for i, r in enumerate(results):
            # judge if have mask.
            if r.masks is None:
                none_index.append(i)
                one_batch_mask[i] = None
            elif list(r.masks.data.shape) == [1, 224, 224]:
                one_batch_mask[i] = r.masks.data  # 1, 224, 224
            else:
                # when mask > 2, just use the first mask.
                # ? sometime will get two type for masks.
                one_batch_mask[i] = r.masks.data[:1, ...]  # 1, 224, 224

        return one_batch_mask, none_index

    def delete_tensor(self, video: torch.tensor, delete_idx: int, next_idx: int):
        """
        delete_tensor, from video, we delete the delete_idx tensor and insert the next_idx tensor.

        Args:
            video (torch.tensor): video tensor for process.
            delete_idx (int): delete tensor index.
            next_idx (int): insert tensor index.

        Returns:
            torch.tensor: deleted and processed video tensor.
        """

        c, t, h, w = video.shape
        left = video[:, :delete_idx, ...]
        right = video[:, delete_idx+1:, ...]
        insert = video[:, next_idx, ...].unsqueeze(dim=1)

        ans = torch.cat([left, insert, right], dim=1)

        # check frame
        assert ans.shape[1] == t
        return ans

    def process_none(self, batch_Dict: dict, none_index: list):
        """
        process_none, where from batch_Dict to instead the None value with next frame tensor (or froward frame tensor).

        Args:
            batch_Dict (dict): batch in Dict, where include the None value when yolo dont work.
            none_index (list): none index list map to batch_Dict, here not use this.

        Returns:
            list: list include the replace value for None value.
        """

        boundary = len(batch_Dict)-1  # 8

        for k, v in batch_Dict.items():
            if v == None:
                if None in list(batch_Dict.values())[k:] and len(set(list(batch_Dict.values())[k:])) == 1:
                    next_idx = k - 1
                else:
                    next_idx = k+1
                    while batch_Dict[next_idx] == None and next_idx < boundary:
                        next_idx += 1

                batch_Dict[k] = batch_Dict[next_idx]

        return list(batch_Dict.values())

    def process_batch(self, batch: torch.Tensor, labels: list):

        b, c, t, h, w = batch.shape

        # for one batch prepare.
        pred_mask_list = []
        pred_keypoint_list = []

        for batch_index in range(b):
            one_batch_mask_Dict, one_mask_none_index = self.get_YOLO_mask_result(
                batch[batch_index])
            one_batch_mask = self.process_none(
                one_batch_mask_Dict, one_mask_none_index)
            one_batch_keypoint_Dict, one_pose_none_index = self.get_YOLO_pose_result(
                batch[batch_index])
            one_batch_keypoint = self.process_none(
                one_batch_keypoint_Dict, one_pose_none_index)

            # none index union check
            # unio_none_list = list(set(one_mask_none_index).union(set(one_pose_none_index)))

            # * if have none index, we copy the next frame to compensate for missing frames.
            # * this method needs to ensure that at least one vaild frame is in the list.
            # if len(unio_none_list) != 0:
            #     for none_idx in unio_none_list:
            #         # prepare next index
            #         if none_idx == t-1: next_idx = t-2
            #         else: next_idx = none_idx + 1

            #         if none_idx in one_batch_mask.keys():
            #             one_batch_mask[none_idx] = one_batch_mask[next_idx]
            #         if none_idx in one_batch_keypoint.keys():
            #             one_batch_keypoint[none_idx] = one_batch_keypoint[next_idx]

            #         # process in raw video and label
            #         process_batch = self.delete_tensor(batch[batch_index], none_idx, next_idx) # c, t, h, w
            #         process_batch_list.append(process_batch) # c, t, h, w
            # else:
            #     process_batch_list.append(batch[batch_index]) # c, t, h, w

            pred_mask_list.append(torch.stack(
                one_batch_mask, dim=1))  # c, t, h, w
            pred_keypoint_list.append(torch.stack(
                one_batch_keypoint, dim=1))  # c, t, keypoint, value

        # return batch, label, mask, keypoint
        return batch, labels, torch.stack(pred_mask_list, dim=0), torch.stack(pred_keypoint_list, dim=0)

    def forward(self, batch, labels):

        b, c, t, h, w, = batch.shape

        # batch, (b, c, t, h, w)
        # label, (b)
        # mask, (b, 1, t, h, w)
        # keypoint, (b, 1, t, 17, 2)
        video, labels, mask, keypoint = self.process_batch(batch, labels)

        # shape check
        assert video.shape == batch.shape
        assert labels.shape[0] == b
        assert mask.shape[2] == t
        assert keypoint[2] == t

        return video, labels, mask, keypoint
