#!/usr/bin/env python
# -*- coding: utf-8 -*-


from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from combustion.nn import MobileNetBlockConfig
from combustion.vision.centernet import CenterNetMixin

from .efficientdet import EfficientDet2d


class MultiLevelHead(nn.Module):
    def __init__(self, in_channels, num_classes, num_repeats: int = 4):
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

        self.cls_head = nn.Sequential(*cls_head)
        self.reg_head = nn.Sequential(*reg_head)

    def forward(self, inputs: List[Tensor]) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        cls_pred = [self.cls_head(x) for x in inputs]
        reg_pred = [self.reg_head(x) for x in inputs]

        # extract centerness
        centerness = [x[..., 0:1, :, :] for x in reg_pred]

        # apply ReLU and reweight
        reg_pred = [F.relu(x[..., 1:, :, :]) * (i - 1 ** 2) for i, x in enumerate(reg_pred)]

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
        self.head = MultiLevelHead(fpn_filters, num_classes, num_repeats)

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
    ) -> Tuple[Tensor, Tuple[Tensor, ...]]:
        r"""Applys postprocessing to create a set of anchorboxes from FCOS predictions."""

        locations, boxes = [], []
        # iterate over each FPN level
        for i, (stride, level_cls, level_reg, level_centerness) in enumerate(zip(strides, cls, reg, centerness)):
            # scale classifications based on centerness
            scaled_cls = level_cls * level_centerness.expand_as(level_cls)

            # get indices of positions that exceed threshold
            positive_locations = (scaled_cls >= threshold).nonzero(as_tuple=False)
            locations.append(positive_locations)
            batch, cls, y, x = positive_locations.split(1, dim=-1)
            level_cls.shape[0]

            # use stride to compute base coodinates within the original image
            # use pred regression to compute l, t, r, b offset
            base = (positive_locations[..., -2:] * stride).repeat(1, 2)
            offset = level_reg[batch, cls, y, x]

            # use original score before centerness scaling
            score = level_cls[batch, cls, y, x]

            # build final bbox of form x1, y1, x2, y2, score, class
            boxes_for_level = torch.cat([base + offset, score, cls], dim=-1)

            # split boxes by batch and apply padding
            _, batch_counts = batch.unique(return_counts=True)
            boxes_for_level = CenterNetMixin.batch_box_target(boxes_for_level.split(batch_counts.tolist()), pad_value)
            boxes.append(boxes_for_level)

        # combine boxes across all FPN levels
        # we might have added a lot of unnecessary padding so reduce if possible
        boxes = torch.cat(boxes, dim=-2)
        boxes = CenterNetMixin.batch_box_target(CenterNetMixin.unbatch_box_target(boxes, pad_value), pad_value)

        return boxes, locations
