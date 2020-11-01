#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from combustion.util import check_dimension, check_dimension_match, check_is_tensor

from .ciou import CompleteIoULoss
from .focal import FocalLossWithLogits


# FCOS uses each FPN level to predict targets of different sizes
# This is the size range used in the paper
DEFAULT_INTEREST_RANGE: Tuple[Tuple[int, int], ...] = (
    (-1, 64),  # stride=8
    (64, 128),  # stirde=16
    (128, 256),  # stride=32
    (256, 512),  # stride=64
    (512, 10000000),  # stride=128
)

IGNORE = -1


class FCOSLoss:
    r"""Implements the loss function and target creation as described in PLACEHOLDER."""

    def __init__(
        self,
        strides: Tuple[int, ...],
        size_targets: Tuple[Tuple[int, int], ...],
        num_classes: int,
        interest_range: Tuple[Tuple[int, int], ...] = DEFAULT_INTEREST_RANGE,
        gamma: float = 2.0,
        alpha: float = 0.5,
        radius: Optional[int] = 1,
        pad_value: float = -1,
    ):
        self.strides = tuple([int(x) for x in strides])
        self.size_targets = tuple([(int(x), int(y)) for x, y in size_targets])
        self.interest_range = tuple([(int(x), int(y)) for x, y in interest_range])
        self.num_classes = int(num_classes)
        self.pad_value = pad_value

        self.pad_value = float(pad_value)
        self.radius = int(radius) if radius is not None else None

        self.cls_criterion = FocalLossWithLogits(gamma, alpha, reduction="none")
        self.reg_criterion = CompleteIoULoss(reduction="none")
        self.centerness_criterion = nn.BCEWithLogitsLoss(reduction="none")

    def __call__(
        self,
        cls_pred: Tuple[Tensor, ...],
        reg_pred: Tuple[Tensor, ...],
        centerness_pred: Tuple[Tensor, ...],
        target_bbox: Tensor,
        target_cls: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        batch_size = target_bbox.shape[0]
        cls_loss, reg_loss, centerness_loss = [], [], []

        for i in range(batch_size):
            cls = [t[i] for t in cls_pred]
            reg = [t[i] for t in reg_pred]
            centerness = [t[i] for t in centerness_pred]
            _target_bbox = self._drop_padding(target_bbox[i])
            _target_cls = self._drop_padding(target_cls[i])
            _cls_loss, _reg_loss, _centerness_loss = self.compute_from_box_target(
                cls, reg, centerness, _target_bbox, _target_cls
            )
            cls_loss.append(_cls_loss)
            reg_loss.append(_reg_loss)
            centerness_loss.append(_centerness_loss)

        cls_loss = sum(cls_loss)
        reg_loss = sum(reg_loss)
        centerness_loss = sum(centerness_loss)
        return cls_loss, reg_loss, centerness_loss

    def _drop_padding(self, x: Tensor) -> Tensor:
        padding = (x == self.pad_value).all(dim=-1)
        return x[~padding]

    def compute_from_box_target(
        self,
        cls_pred: Tuple[Tensor, ...],
        reg_pred: Tuple[Tensor, ...],
        centerness_pred: Tuple[Tensor, ...],
        target_bbox: Tensor,
        target_cls: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        fcos_targets = self.create_targets(target_bbox, target_cls)
        return self.compute_from_fcos_target(cls_pred, reg_pred, centerness_pred, fcos_targets)

    def compute_from_fcos_target(
        self,
        cls_pred: Tuple[Tensor, ...],
        reg_pred: Tuple[Tensor, ...],
        centerness_pred: Tuple[Tensor, ...],
        fcos_target: Tuple[Tuple[Tensor, Tensor, Tensor], ...],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        cls_loss = [self.cls_criterion(pred, true) for pred, (true, _, _) in zip(cls_pred, fcos_target)]
        centerness_loss = [
            self.centerness_criterion(pred, true) for pred, (_, _, true) in zip(centerness_pred, fcos_target)
        ]

        reg_loss = [
            self.reg_criterion(pred.view(4, -1).permute(1, 0), true.view(4, -1).permute(1, 0))
            .view(pred.shape[1:])
            .unsqueeze_(0)
            for pred, (_, true, _) in zip(reg_pred, fcos_target)
        ]

        cls_loss = sum([x.sum() for x in cls_loss])
        reg_loss = sum([x.sum() for x in reg_loss])
        centerness_loss = sum([x.sum() for x in centerness_loss])

        return cls_loss, reg_loss, centerness_loss

    def create_targets(
        self,
        bbox: Tensor,
        cls: Tensor,
    ) -> Tuple[Tuple[Tensor, Tensor, Tensor], ...]:
        class_targets, reg_targets, centerness_targets = [], [], []
        for irange, stride, size_target in zip(self.interest_range, self.strides, self.size_targets):
            _cls, _reg, _centerness = FCOSLoss.create_target_for_level(
                bbox, cls, self.num_classes, stride, size_target, irange, self.radius
            )
            class_targets.append(_cls)
            reg_targets.append(_reg)
            centerness_targets.append(_centerness)

        return tuple(
            [(cls, reg, centerness) for cls, reg, centerness in zip(class_targets, reg_targets, centerness_targets)]
        )

    @staticmethod
    def create_target_for_level(
        bbox: Tensor,
        cls: Tensor,
        num_classes: int,
        stride: int,
        size_target: Tuple[int, int],
        interest_range: Tuple[int, int],
        center_radius: Optional[int] = None,
    ) -> Tensor:
        # get bbox locations within feature map after stride is applied
        bbox_stride = bbox.floor_divide(stride)

        # build regression target
        reg_target = FCOSLoss.create_regression_target(bbox_stride, stride, size_target)

        # use the regression targets to determine boxes of interest for this level
        # is of interest if lower_bound <= max(l, r, t, b) <= upper_bound
        max_size = reg_target.view(reg_target.shape[0], -1).max(dim=-1).values
        lower_bound, upper_bound = interest_range
        is_box_of_interest = (max_size >= lower_bound).logical_and_(max_size <= upper_bound)

        # get mask of valid locations within each box and apply boxes_of_interest filter
        mask = FCOSLoss.bbox_to_mask(bbox, stride, size_target, center_radius)
        mask[~is_box_of_interest] = False

        # build classification target
        cls_target = FCOSLoss.create_classification_target(bbox, cls, mask, num_classes, size_target)

        # apply mask to regression target and take per pixel maximum for all boxes
        reg_target[~mask[..., None, :, :].expand_as(reg_target)] = IGNORE
        reg_target = reg_target.max(dim=0).values

        centerness_target = FCOSLoss.compute_centerness_targets(reg_target)
        centerness_target[~mask.any(dim=-3, keepdim=True)] = IGNORE

        return cls_target, reg_target, centerness_target

    @staticmethod
    def bbox_to_mask(
        bbox: Tensor, stride: int, size_target: Tuple[int, int], center_radius: Optional[float] = None
    ) -> Tensor:
        check_is_tensor(bbox, "bbox")
        check_dimension(bbox, -1, 4, "bbox")

        # create empty masks
        num_boxes = bbox.shape[-2]
        h = torch.arange(size_target[0], dtype=bbox.dtype, device=bbox.device)
        w = torch.arange(size_target[1], dtype=bbox.dtype, device=bbox.device)
        mask = torch.stack(torch.meshgrid(h, w), 0).unsqueeze_(0).expand(num_boxes, -1, -1, -1)

        # get edge coordinates of each box based on whole box or center sampled
        lower_bound = bbox[..., :2]
        upper_bound = bbox[..., 2:]
        if center_radius is not None:
            # update bounds according to radius from center
            center = (bbox[..., :2] + bbox[..., 2:]).floor_divide_(2)
            offset = torch.tensor([stride, stride], device=bbox.device, dtype=center.dtype).mul_(center_radius)
            lower_bound = center - offset[None]
            upper_bound = center + offset[None]

        # x1y1 to h1w1, add h/w dimensions, convert to strided coords
        lower_bound = lower_bound[..., (1, 0), None, None].floor_divide_(stride)
        upper_bound = upper_bound[..., (1, 0), None, None].floor_divide_(stride)

        # use edge coordinates to create a binary mask
        mask = (mask > lower_bound).logical_and_(mask < upper_bound).all(dim=-3)
        return mask

    @staticmethod
    def create_regression_target(
        bbox: Tensor,
        stride: int,
        size_target: Tuple[int, int],
    ) -> Tensor:
        check_is_tensor(bbox, "bbox")
        check_dimension(bbox, -1, 4, "bbox")

        # create starting grid
        num_boxes = bbox.shape[-2]
        h = torch.arange(size_target[0], dtype=bbox.dtype, device=bbox.device)
        w = torch.arange(size_target[1], dtype=bbox.dtype, device=bbox.device)
        grid = torch.meshgrid(h, w)
        grid = torch.stack([grid[1], grid[0]], dim=0).unsqueeze_(0).repeat(num_boxes, 2, 1, 1)
        grid.mul_(stride)

        # compute distance to box edges relative to each grid location
        grid[..., :2, :, :].sub_(bbox[..., :2, None, None])
        grid[..., 2:, :, :].neg_().add_(bbox[..., 2:, None, None])
        return grid

    @staticmethod
    def create_classification_target(
        bbox: Tensor,
        cls: Tensor,
        mask: Tensor,
        num_classes: int,
        size_target: Tuple[int, int],
    ) -> Tensor:
        check_is_tensor(bbox, "bbox")
        check_is_tensor(cls, "cls")
        check_is_tensor(mask, "mask")
        check_dimension_match(bbox, cls, -2, "bbox", "cls")
        check_dimension_match(bbox, mask, 0, "bbox", "mask")
        check_dimension(bbox, -1, 4, "bbox")
        check_dimension(cls, -1, 1, "cls")

        target = torch.zeros(num_classes, *mask.shape[-2:], device=mask.device, dtype=torch.float)

        box_id, h, w = mask.nonzero(as_tuple=True)
        class_id = cls[box_id, 0]
        target[class_id, h, w] = 1.0
        return target

    @staticmethod
    def compute_centerness_targets(reg_targets: Tensor) -> Tensor:
        r"""Computes centerness targets given a 2D map of regression targets.

        Under FCOS, a target regression map is created for each FPN level. Any map location
        that lies within a ground truth bounding box is assigned a regression target based on
        the left, right, top, and bottom distance from that location to the edges of the ground
        truth box.

        .. image:: ./fcos_target.png
            :width: 200px
            :align: center
            :height: 600px
            :alt: FCOS Centerness Target

        For each of these locations with regression targets :math:`l^*, r^*, t^*, b^*`,
        a "centerness" target is created as follows:

        .. math::
            centerness = \sqrt{\frac{\min(l^*, r*^}{\max(l^*, r*^} \times \frac{\min(t^*, b*^}{\max(t^*, b*^}}

        Args:
            reg_targets (:class:`torch.Tensor`):
                Ground truth regression featuremap in form :math:`x_1, y_1, x_2, y_2`.

        Shapes:
            * ``reg_targets`` - :math:`(..., 4, H, W)`
            * Output - :math:`(..., 1, H, W)`
        """
        check_is_tensor(reg_targets, "reg_targets")
        check_dimension(reg_targets, -3, 4, "reg_targets")

        left_right = reg_targets[(0, 2), ...].float()
        top_bottom = reg_targets[(1, 3), ...].float()

        lr_min = left_right.min(dim=-3).values.clamp_min_(0)
        lr_max = left_right.max(dim=-3).values.clamp_min_(1)
        tb_min = top_bottom.min(dim=-3).values.clamp_min_(0)
        tb_max = top_bottom.max(dim=-3).values.clamp_min_(1)

        centerness_lr = lr_min.true_divide_(lr_max)
        centerness_tb = tb_min.true_divide_(tb_max)
        centerness = centerness_lr.mul_(centerness_tb).sqrt_().unsqueeze_(-3)

        assert centerness.shape[-2:] == reg_targets.shape[-2:]
        assert centerness.shape[-3] == 1
        assert centerness.ndim == reg_targets.ndim
        return centerness
