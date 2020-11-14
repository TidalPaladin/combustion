#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from torch import Tensor
import kornia
import random
from kornia.augmentation import RandomCrop, RandomErasing

from typing import Tuple, Optional, Union, List
from combustion.vision import AnchorsToPoints
from combustion.vision.centernet import CenterNetMixin
from albumentations.core.composition import Compose, BboxParams

from medcog_classifier.mixins import MalignancyMixin
from medcog_preprocessing.data import Embeddings
from .filter import filter_box_target, get_positional_emb

class TrainingTransform:
    r"""Configurable combination of transforms for use at training time.

    Args:
        num_classes (int):
            Number of classes in the target labels

        downsample (int):
            CenterNet heatmap will be produced at ``1/downsample`` the input resolution.

        return_heatmap (bool):
            If ``True``, return a CenterNet heatmap

        keep_classes (list of str):
            An optional list of class names to keep

    Keyword args:
        A set of Albumentations transforms to compose. Transformations given here are applied to
        the input image and target before converstion to CenterNet labels.

    Shape
        * ``img`` - :math:`(C, H, W)`
        * ``target`` - :math:`(N, 5)` where ``target`` is of the form ``(x1, y1, x2, y2, type, malig)``
        * Output target - :math:`(C_t + 4, H / d, W / d)` where :math:`C_t` is the number of target classes
          and :math:`d` is ``downsample``.
    """
    def __init__(
        self, 
        num_classes: int,
        downsample: int,
        return_heatmap: bool = False,
        keep_classes: Optional[List[str]] = None,
        invert: float = 0.0,
        size_limit: Optional[int] = None,
        **kwargs
    ):
        self.downsample = downsample
        self.size_limit = size_limit
        self.num_classes = num_classes
        self.invert = float(invert)
        self.transforms = Compose(
            [v for v in kwargs.values()],
            bbox_params=BboxParams(format="pascal_voc", label_fields=["types", "malig"])
        )
        self.keep_classes = keep_classes
        if keep_classes is not None:
            self.emb = Embeddings()
            self.emb.remap_types(keep_classes)

    def __call__(self, img: Tensor, target: List[Tensor]) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]:
        target = target[0]
        if not img.is_floating_point():
            img = img.float().div_(2**16).add_(0.5).clamp_(min=0, max=1)

        #img = torch.cat([img, get_positional_emb(img)], dim=0)

        # random image inversion
        if random.random() < self.invert:
            img[0].neg_().add_(1)

        # convert img to channels last numpy array for albumentations
        img = img.detach().permute(1, 2, 0).numpy()

        # filter class types if needed
        if self.keep_classes is not None:
            target = filter_box_target(target[..., (0, 1, 2, 3, -1, -2)], self.keep_classes, self.emb)
            target = target[..., (0, 1, 2, 3, -1, -2)].contiguous()

        # drop padding if present
        padding = (target[..., 4:5] == -1).all(dim=-1)
        target = target[~padding]

        # split target into bbox type and malig components
        bbox, types, malig = CenterNetMixin.split_box_target(target, split_label=True)
        assert bbox.shape[-1] == 4
        assert types.shape[-1] == 1
        assert malig.shape[-1] == 1

        # convert targets to numpy
        bbox, types, malig = [x.numpy() for x in (bbox, types, malig)]
        
        # apply albumentations tranasform and extract results
        output = self.transforms(image=img, bboxes=bbox, types=types, malig=malig)
        img = torch.from_numpy(output["image"]).permute(2, 0, 1).contiguous().float()
        bbox = torch.as_tensor(output["bboxes"], dtype=torch.float)
        types = torch.as_tensor(output["types"], dtype=torch.float)
        malig = torch.as_tensor(output["malig"], dtype=torch.float)

        # ensure all bounding boxes have a nonzero area
        if bbox.numel():
            valid_boxes = (bbox[..., 0] < bbox[..., 2]).logical_and_(bbox[..., 1] < bbox[..., 3])
            if self.size_limit is not None:
                above_size_limit = (bbox[..., 2:4] - bbox[..., :2] > self.size_limit).all(dim=-1)
                valid_boxes.logical_and_(above_size_limit)
            bbox = bbox[valid_boxes]
            types = types[valid_boxes]
            malig = malig[valid_boxes]

        # if no boxes in crop
        if not bbox.numel():
            bbox = bbox.view(0, 4)
            types = types.view(0, 1)
            malig = malig.view(0, 1)

        malig.fill_(-1)

        # return bbox target if desired
        target = CenterNetMixin.combine_box_target(bbox, types, malig)
        padded_target = torch.empty(200, target.shape[-1]).fill_(-1)
        padded_target[:target.shape[0], ...] = target
        assert padded_target.shape[-1] == 6
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


        target = CenterNetMixin.batch_box_target([x[1] for x in examples])
        return img, target
