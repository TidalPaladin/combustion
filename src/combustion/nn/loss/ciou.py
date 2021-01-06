#!/usr/bin/env python
# -*- coding: utf-8 -*-

from math import pi

import torch
import torch.nn as nn
from torch import Tensor

from combustion.util import check_dimension, check_is_tensor, check_shapes_match


def absolute_to_center_delta(bbox: Tensor) -> Tensor:
    _ = bbox.view(-1, 2, 2)
    _ = _ - _[..., 0:1, :]
    return _.view_as(bbox)


def complete_iou_loss(inputs: Tensor, targets: Tensor, reduction: str = "mean", absolute: bool = False) -> Tensor:
    # validation
    check_is_tensor(inputs, "inputs")
    check_is_tensor(targets, "targets")
    check_dimension(inputs, -1, 4, "inputs")
    check_dimension(targets, -1, 4, "targets")
    check_shapes_match(inputs, targets, "inputs", "targets")

    inputs = inputs.float().clone()
    targets = targets.float().clone()

    # convert absolute coordinates to distance from box center if needed
    if absolute:
        inputs = absolute_to_center_delta(inputs)
        targets = absolute_to_center_delta(targets)
    else:
        # x1, y1 are negative deltas relative to center
        inputs[..., :2] = inputs[..., :2].neg()
        targets[..., :2] = targets[..., :2].neg()

    # compute euclidean distance between pred and true box centers
    pred_size = (inputs[..., 2:] - inputs[..., :2]).clamp_min(1)
    target_size = (targets[..., 2:] - targets[..., :2]).clamp_min(1)
    pred_center = pred_size.div(2).add(inputs[..., :2])
    target_center = target_size.div(2).add(targets[..., :2])
    euclidean_dist_squared = (pred_center - target_center).pow(2).sum(dim=-1)

    # compute c, the diagonal length of smallest box enclosing pred and true
    min_coords = torch.min(inputs[..., :2], targets[..., :2])
    max_coords = torch.max(inputs[..., 2:], targets[..., 2:])
    c_squared = (max_coords - min_coords).pow(2).sum(dim=-1)

    # compute diou
    diou = euclidean_dist_squared / c_squared

    # compute vanilla IoU
    pred_area = pred_size[..., 0] * pred_size[..., 1]
    target_area = target_size[..., 0] * target_size[..., 1]
    lt = torch.max(inputs[..., :2], targets[..., :2])
    rb = torch.min(inputs[..., 2:], targets[..., 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    iou = inter / (pred_area + target_area - inter).clamp_min(1e-9)

    # compute v, which measure aspect ratio consistency
    pred_w, pred_h = pred_size[..., 0], pred_size[..., 1].clamp_min(1e-4)
    target_w, target_h = target_size[..., 0], target_size[..., 1].clamp_min(1e-5)
    _ = torch.atan(target_w / target_h) - torch.atan(pred_w / pred_h)
    v = 4 / pi ** 2 * _.pow(2)

    # compute alpha, the tradeoff parameter
    alpha = v / ((1 - iou) + v).clamp_min(1e-5)

    # compute the final ciou loss
    loss = 1 - iou + diou + alpha * v

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss
    else:
        raise ValueError(f"Unknown reduction {reduction}")


class CompleteIoULoss(nn.Module):
    r"""Implements Complete IoU loss as described in `Distance-IoU Loss`_.
    This is a bounding box regression loss.

    Complete IoU loss uses several criteria for comparing bounding boxes, including
        1. Box IoU
        2. Euclidean distance of box centers
        3. Similarity of box aspect ratios

    Args:
        inputs (:class:`torch.Tensor`):
            Predicted boxes in form ``x1, y1, x2, y2``

        targets (:class:`torch.Tensor`):
            Target boxes in form ``x1, y1, x2, y2``

        reduction (str):
            How to reduce the loss to a scalar

    Shapes:
        * ``inputs`` - :math:`(*, N, 4)`
        * ``targets`` - :math:`(*, N, 4)`

    .. _Distance-IoU Loss:
        https://arxiv.org/abs/1911.08287
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        return complete_iou_loss(inputs, targets, self.reduction)
