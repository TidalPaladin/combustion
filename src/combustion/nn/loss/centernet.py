#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional, Tuple

from torch import Tensor
from torch.nn import L1Loss, SmoothL1Loss

from .focal import FocalLossWithLogits


class CenterNetLoss:
    def __init__(
        self,
        gamma: float = 2.0,
        pos_weight: float = 0.5,
        label_smoothing: Optional[float] = None,
        reduction: str = "mean",
        smooth: bool = True,
    ):
        self.reduction = reduction
        self.cls_criterion = FocalLossWithLogits(gamma, pos_weight, label_smoothing, reduction="none")
        self.loc_criterion = SmoothL1Loss(reduction="none") if smooth else L1Loss(reduction="none")

    def __call__(self, inputs: Tensor, targets: Tensor) -> Tuple[Tensor, Tensor]:
        assert inputs.shape == targets.shape

        # split input/target classes / regressions
        pred_regression = inputs[..., -4:, :, :]
        pred_class = inputs[..., :-4, :, :]
        tar_regression = targets[..., -4:, :, :]
        tar_class = targets[..., :-4, :, :]

        # compute classification/regression loss
        cls_loss = self.cls_criterion(pred_class, tar_class)
        reg_loss = self.loc_criterion(pred_regression, tar_regression)

        # zero out regression loss for locations that were not box centers
        box_indices = (tar_class == 1.0).max(dim=-3, keepdim=True).values.expand_as(reg_loss)
        reg_loss[~box_indices] = 0

        if self.reduction == "mean":
            num_boxes = box_indices.sum().div_(4).clamp_(min=1)
            cls_loss = cls_loss.sum().div_(num_boxes)
            reg_loss = reg_loss.sum().div_(num_boxes)
        elif self.reduction == "sum":
            cls_loss = cls_loss.sum()
            reg_loss = reg_loss.sum()

        return cls_loss, reg_loss
