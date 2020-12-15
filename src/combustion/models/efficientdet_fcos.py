#!/usr/bin/env python
# -*- coding: utf-8 -*-


from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.ops import batched_nms

from combustion.nn import MobileNetBlockConfig
from combustion.vision import batch_box_target

from .efficientdet import EfficientDet2d


class MultiLevelHead(nn.Module):
    def __init__(self, in_channels, num_classes, strides: Tuple[int, ...], num_repeats: int = 4):
        super().__init__()
        cls_head, reg_head = [], []

        for i in range(num_repeats):
            cls_out_channels = in_channels if i < num_repeats - 1 else num_classes
            reg_out_channels = in_channels if i < num_repeats - 1 else 5
            is_last = i == num_repeats - 1
            cls = self._get_repeat(in_channels, cls_out_channels, is_last=is_last)
            reg = self._get_repeat(in_channels, reg_out_channels, is_last=is_last)
            cls_head.append(cls)
            reg_head.append(reg)

        self.strides = tuple([int(x) for x in strides])
        self.cls_head = nn.Sequential(*cls_head)
        self.reg_head = nn.Sequential(*reg_head)

    def forward(self, inputs: List[Tensor]) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        cls_pred = [self.cls_head(x) for x in inputs]
        reg_pred = [self.reg_head(x) for x in inputs]

        # extract centerness
        centerness = [x[..., 0:1, :, :] for x in reg_pred]

        # apply ReLU and reweight
        reg_pred = [F.relu(x[..., 1:, :, :]) * s for s, x in zip(self.strides, reg_pred)]

        return cls_pred, reg_pred, centerness

    def _get_repeat(
        self, in_channels: int, out_channels: int, act: nn.Module = nn.Hardswish(), is_last: bool = False
    ) -> nn.Module:
        layers = [
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False, groups=in_channels),
            nn.Conv2d(in_channels, out_channels, 1, bias=is_last),
        ]

        if not is_last:
            layers += [nn.BatchNorm2d(in_channels), act]

        return nn.Sequential(*layers)


class EfficientDetFCOS(EfficientDet2d):
    def __init__(
        self,
        num_classes: int,
        block_configs: List[MobileNetBlockConfig],
        strides: List[int] = [8, 16, 32, 64, 128],
        fpn_levels: List[int] = [3, 5, 7, 8, 9],
        fpn_filters: int = 64,
        fpn_repeats: int = 3,
        width_coeff: float = 1.0,
        depth_coeff: float = 1.0,
        width_divisor: float = 8.0,
        min_width: Optional[int] = None,
        stem: Optional[nn.Module] = None,
        fpn_kwargs: dict = {},
        head_repeats: Optional[int] = None,
    ):
        super().__init__(
            block_configs,
            fpn_levels,
            fpn_filters,
            fpn_repeats,
            width_coeff,
            depth_coeff,
            width_divisor,
            min_width,
            stem,
            nn.Identity(),
            fpn_kwargs,
        )

        if head_repeats is None:
            if hasattr(self, "compound_coeff"):
                num_repeats = 3 + self.compound_coeff // 3
            else:
                num_repeats = 3
        else:
            num_repeats = head_repeats
        self.strides = strides
        self.head = MultiLevelHead(fpn_filters, num_classes, strides, num_repeats)

    def forward(self, inputs: Tensor) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        r"""Runs the entire EfficientDet model, including stem, body, and head.
        If no head was supplied, the output of :func:`extract_features` will be returned.
        Otherwise, the output of the given head will be returned.

        .. note::
            The returned output will always be a list of tensors. If a custom head is given
            and it returns a single tensor, that tensor will be wrapped in a list before
            being returned.

        Args:
            inputs (:class:`torch.Tensor`):
                Model inputs
        """
        output = self.extract_features(inputs)
        output = self.head(output)
        return output

    @classmethod
    def from_predefined(cls, compound_coeff: int, num_classes: int, **kwargs) -> "EfficientDetFCOS":
        r"""Creates an EfficientDet model using one of the parameterizations defined in the
        `EfficientDet paper`_.

        Args:
            compound_coeff (int):
                Compound scaling parameter :math:`\phi`. For example, to construct EfficientDet-D0, set
                ``compound_coeff=0``.

            **kwargs:
                Additional parameters/overrides for model constructor.

        .. _EfficientNet paper:
            https://arxiv.org/abs/1905.11946
        """
        # from paper
        alpha = 1.2
        beta = 1.1
        width_divisor = 8.0

        depth_coeff = alpha ** compound_coeff
        width_coeff = beta ** compound_coeff

        strides = [8, 16, 32, 64, 128]
        fpn_filters = int(64 * 1.35 ** compound_coeff)
        fpn_repeats = 3 + compound_coeff
        fpn_levels = [3, 5, 7, 8, 9]

        head_repeats = 4 + compound_coeff // 3

        final_kwargs = {
            "num_classes": num_classes,
            "block_configs": cls.DEFAULT_BLOCKS,
            "width_coeff": width_coeff,
            "depth_coeff": depth_coeff,
            "width_divisor": width_divisor,
            "strides": strides,
            "fpn_filters": fpn_filters,
            "fpn_repeats": fpn_repeats,
            "fpn_levels": fpn_levels,
            "head_repeats": head_repeats,
        }
        final_kwargs.update(kwargs)
        result = cls(**final_kwargs)
        result.compound_coeff = compound_coeff
        return result

    @staticmethod
    def create_boxes(
        cls: Tuple[Tensor, ...],
        reg: Tuple[Tensor, ...],
        centerness: Tuple[Tensor, ...],
        strides: Tuple[int, ...],
        threshold: float = 0.05,
        pad_value: float = -1,
        from_logits: bool = False,
        nms_threshold: Optional[float] = 0.5,
        use_raw_score: bool = False,
        max_boxes: Optional[int] = None,
    ) -> Tuple[Tensor, Tuple[Tensor, ...]]:
        r"""Applys postprocessing to create a set of anchorboxes from FCOS predictions."""
        _ = [x * strides[0] for x in cls[0].shape[-2:]]
        y_lim, x_lim = _

        batch_idx, boxes = [], []

        batch_size = cls[0].shape[0]
        num_classes = cls[0].shape[1]

        # iterate over each FPN level
        for i, (stride, level_cls, level_reg, level_centerness) in enumerate(zip(strides, cls, reg, centerness)):

            if from_logits:
                level_cls = torch.sigmoid(level_cls)
                level_centerness = torch.sigmoid(level_centerness)

            # scale classifications based on centerness
            scaled_score = (level_cls * level_centerness.expand_as(level_cls)).sqrt_()

            # get indices of positions that exceed threshold
            positive_locations = (level_cls >= threshold).nonzero(as_tuple=False)

            if not positive_locations.numel():
                continue

            # extract coordinates of positive predictions and drop scores for negative predictions
            batch, cls, y, x = positive_locations.split(1, dim=-1)
            raw_score = level_cls[batch, cls, y, x]
            scaled_score = scaled_score[batch, cls, y, x]

            # use stride to compute base coodinates within the original image
            # use pred regression to compute l, t, r, b offset
            base = (positive_locations[..., (-1, -2)] * stride).add_(int(stride / 2)).repeat(1, 2)
            offset = level_reg[batch, :, y, x].view_as(base)
            offset[..., :2].neg_()

            # compute final regressions and clamp to lie within image_size
            coords = (base + offset).clamp_min_(0)
            coords[..., 2].clamp_max_(x_lim)
            coords[..., 3].clamp_max_(y_lim)

            # record the boxes and box -> batch mapping
            boxes.append(torch.cat([coords, raw_score, scaled_score, cls], dim=-1))
            batch_idx.append(batch)

        # combine boxes across all FPN levels
        if boxes:
            boxes = torch.cat(boxes, dim=-2)
            batch_idx = torch.cat(batch_idx, dim=-2)
        else:
            boxes = coords.new_empty(batch_size, 0, 6)
            return boxes, []

        scaled_score = boxes[..., -2]
        if max_boxes is not None:
            keep = scaled_score.argsort(descending=True)[:max_boxes]
            boxes = boxes[keep]
            batch_idx = batch_idx[keep]

        # apply NMS to boxes
        if nms_threshold is not None:
            coords = boxes[..., :4]
            raw_score = boxes[..., -3, None]
            scaled_score = boxes[..., -2, None]
            cls = boxes[..., -1, None]

            # torchvision NMS cant do batches of images, but it can separate based on class id
            # create a new "class id" that distinguishes batch and class
            idx = (batch_idx * num_classes + cls.view_as(batch_idx)).view(-1).long()
            keep = batched_nms(coords.float(), scaled_score.view(-1), idx, nms_threshold)
            boxes = boxes[keep, :]
            batch_idx = batch_idx[keep, :]

        # create final box using raw or centerness adjusted score as specified
        if use_raw_score:
            boxes = boxes[..., (0, 1, 2, 3, 4, 6)]
        else:
            boxes = boxes[..., (0, 1, 2, 3, 5, 6)]

        # pack boxes into a padded batch
        boxes = [boxes[(batch_idx == i).view(-1), :] for i in range(batch_size)]
        boxes = batch_box_target(boxes, pad_value)

        return boxes, None

    @staticmethod
    def reduce_heatmap(heatmap: Tuple[Tensor, ...]) -> Tensor:
        top = heatmap[0]
        for i in range(len(heatmap) - 1):
            level = heatmap[i + 1]
            top = torch.max(top, F.interpolate(level, top.shape[-2:], mode="nearest"))
        return top
