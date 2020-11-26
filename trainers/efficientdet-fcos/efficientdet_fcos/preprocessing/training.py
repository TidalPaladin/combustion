#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import List, Tuple, Union

import numpy as np
import torch
from albumentations.core.composition import BboxParams, Compose
from torch import Tensor

from combustion.vision import batch_box_target, combine_box_target


class TrainingTransform:
    r"""Configurable combination of transforms for use at training time.

    Args:
        num_classes (int):
            Number of classes in the target labels

    Keyword args:
        A set of Albumentations transforms to compose. Transformations given here are applied to
        the input image and target before converstion to CenterNet labels.

    Shape
        * ``img`` - :math:`(C, H, W)`
        * ``target`` - :math:`(N, 5)` where ``target`` is of the form ``(x1, y1, x2, y2, type, malig)``
        * Output target - :math:`(C_t + 4, H / d, W / d)` where :math:`C_t` is the number of target classes
          and :math:`d` is ``downsample``.
    """

    def __init__(self, **kwargs):
        self.transforms = Compose(
            [v for v in kwargs.values()], bbox_params=BboxParams(format="coco", label_fields=["types"])
        )

    def __call__(
        self, img: Tensor, target: List[Tensor]
    ) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]:
        # convert img to channels last numpy array for albumentations
        img = np.array(img)
        bbox = torch.tensor([x["bbox"] for x in target])
        types = torch.tensor([x["category_id"] for x in target]).unsqueeze_(-1)

        assert not bbox.numel() or bbox.shape[-1] == 4
        assert not types.numel() or types.shape[-1] == 1

        # ensure nonzero box size
        bbox[..., 2:].clamp_min_(1e-3)

        # convert targets to numpy
        bbox, types = [x.numpy() for x in (bbox, types)]

        # apply albumentations tranasform and extract results
        output = self.transforms(image=img, bboxes=bbox, types=types)
        img = torch.from_numpy(output["image"]).permute(2, 0, 1).contiguous().float().div_(255)
        bbox = torch.as_tensor(output["bboxes"], dtype=torch.float)
        types = torch.as_tensor(output["types"], dtype=torch.float)

        # convert from coco format (x1, y1, width, height) to (x1, y1, x2, y2)
        bbox[..., -2:].add_(bbox[..., :2])
        bbox.round_()

        # ensure all bounding boxes have a nonzero area
        if bbox.numel():
            valid_boxes = (bbox[..., 0] < bbox[..., 2]).logical_and_(bbox[..., 1] < bbox[..., 3])
            bbox = bbox[valid_boxes]
            types = types[valid_boxes]

        # if no boxes in crop
        if not bbox.numel():
            bbox = bbox.view(0, 4)
            types = types.view(0, 1)

        # return bbox target if desired
        target = combine_box_target(bbox, types)
        padded_target = torch.empty(200, target.shape[-1]).fill_(-1)
        padded_target[: target.shape[0], ...] = target
        assert padded_target.shape[-1] == 5
        return img, padded_target


class Collate:
    def __call__(self, examples):
        batch_size, channels = len(examples), examples[0][0].shape[0]
        max_h = max(x[0].shape[-2] for x in examples)
        max_w = max(x[0].shape[-1] for x in examples)
        img = torch.stack([x[0] for x in examples])

        img = examples[0][0].new_empty(batch_size, channels, max_h, max_w)
        for batch_idx, sub_img in enumerate(examples):
            h, w = sub_img.shape[-2:]
            img[i, :, :h, :w]

        target = batch_box_target([x[1] for x in examples])
        return img, target
