#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional, Tuple

import torch.nn as nn
from torch import Tensor
from torch.nn import L1Loss, SmoothL1Loss

from .focal import FocalLossWithLogits


class CenterNetLoss(nn.Module):
    r"""The loss function used for CenterNet and similar networks, as described
    in the paper `Objects as Points`_.

    Args:

        gamma (float):
            The focusing parameter :math:`\gamma`. Must be non-negative. Note that
            this parameter is referred to as :math:`\alpha` in `Objects as Points`_
            and :math:`\gamma` in the focal loss literature.

        pos_weight (float, optional):
            The positive weight coefficient :math:`\alpha` to use on
            the positive examples. Must be non-negative. Note that
            this parameter is referred to as :math:`\beta` in `Objects as Points`_
            and :math:`\alpha` in the focal loss literature.

        label_smoothing (float, optional):
            Float in [0, 1]. When 0, no smoothing occurs. When positive, the binary
            ground truth labels are clamped to :math:`[p, 1-p]`.

        reduction (str, optional):
            Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed.
            Default: ``'mean'``

        smooth (bool, optional):
            If true, use a smooth L1 loss to compute regression losses. Default ``True``.

    Returns:
        Tuple of tensors giving the classification and regression losses respectively.
        If ``reduction='none'`` the output tensors will be the same shape as inputs, otherwise
        scalar tensors will be returned.

    Shape
        - Inputs: :math:`(*, N+4, H, W)` where :math:`*` means an optional batch dimension
          and :math:`N` is the number of classes. Indices :math:`N+1, N+2` should give the
          :math:`x, y` regression offsets, while indices :math:`N+3, N+4` should give the
          height and width regressions.
        - Targets: Same shape as input.


    .. _Objects as Points:
        https://arxiv.org/abs/1904.07850

    """

    def __init__(
        self,
        gamma: float = 2.0,
        pos_weight: float = 4.0,
        label_smoothing: Optional[float] = None,
        reduction: str = "mean",
        smooth: bool = True,
    ):
        super(CenterNetLoss, self).__init__()
        self.reduction = reduction
        self.cls_criterion = FocalLossWithLogits(gamma, label_smoothing=label_smoothing, reduction="none")
        self.loc_criterion = SmoothL1Loss(reduction="none") if smooth else L1Loss(reduction="none")
        self.pos_weight = pos_weight

    def forward(self, inputs: Tensor, targets: Tensor) -> Tuple[Tensor, Tensor]:
        assert inputs.shape == targets.shape

        # split input/target classes / regressions
        pred_regression = inputs[..., -4:, :, :]
        pred_class = inputs[..., :-4, :, :]
        tar_regression = targets[..., -4:, :, :]
        tar_class = targets[..., :-4, :, :]

        # compute classification/regression loss
        cls_loss = self.cls_criterion(pred_class, tar_class.floor_divide(1.0))
        reg_loss = self.loc_criterion(pred_regression, tar_regression)

        # apply positive example weight to focal loss
        positive_indices = tar_class == 1.0
        weight = (1.0 - tar_class[~positive_indices]).pow_(self.pos_weight)
        positive_examples = cls_loss[~positive_indices]
        positive_examples.mul_(weight)

        # zero out regression loss for locations that were not box centers
        box_indices = positive_indices.max(dim=-3, keepdim=True).values.expand_as(reg_loss)
        reg_loss[~box_indices] = 0

        if self.reduction == "mean":
            num_boxes = box_indices.sum().floor_divide_(4).clamp_(min=1)
            cls_loss = cls_loss.sum().div_(num_boxes)
            reg_loss = reg_loss.sum().div_(num_boxes)
        elif self.reduction == "sum":
            cls_loss = cls_loss.sum()
            reg_loss = reg_loss.sum()

        return cls_loss, reg_loss
