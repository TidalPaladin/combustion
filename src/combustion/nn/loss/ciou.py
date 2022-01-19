#!/usr/bin/env python
# -*- coding: utf-8 -*-

from math import pi

import torch
from typing import Optional
import torch.nn as nn
from torch import Tensor

from combustion.util import check_dimension, check_is_tensor, check_shapes_match

EPS = 1e-12

def cxcy(x: Tensor) -> Tensor:
    return x[..., :2]


def wh(x: Tensor) -> Tensor:
    return x[..., 2:]

def absolute_to_center_delta(bbox: Tensor) -> Tensor:
    _ = bbox.view(-1, 2, 2)
    _ = _ - _[..., 0:1, :]
    return _.view_as(bbox)


def xyxy_to_cxcy(bbox: Tensor) -> Tensor:
    cxcy = (bbox[..., :2] + bbox[..., 2:]) / 2
    wh = bbox[..., 2:] - bbox[..., :2]
    return torch.cat([cxcy, wh], dim=-1)


def cxcy_to_xyxy(bbox: Tensor) -> Tensor:
    x1y1 = cxcy(bbox) - wh(bbox) / 2
    x2y2 = cxcy(bbox) + wh(bbox) / 2
    return torch.cat([x1y1, x2y2], dim=-1)


def compute_box_size(bbox: Tensor) -> Tensor:
    return bbox[..., -2:]


def compute_center_distance(bbox1: Tensor, bbox2: Tensor, norm: Optional[int] = None) -> Tensor:
    result = bbox2[..., :2] - bbox1[..., :2]
    if norm is not None:
        result = result.norm(p=norm, dim=-1)
    return result


def compute_enclosing_diagonal(bbox1: Tensor, bbox2: Tensor) -> Tensor:
    # compute c, the diagonal length of smallest box enclosing pred and true
    bbox1 = cxcy_to_xyxy(bbox1)
    bbox2 = cxcy_to_xyxy(bbox2)
    min_coords = torch.min(bbox1[..., :2], bbox2[..., :2])
    max_coords = torch.max(bbox1[..., 2:], bbox2[..., 2:])
    diag = (max_coords - min_coords).pow(2).sum(dim=-1).sqrt()
    return diag


def compute_iou(bbox1: Tensor, bbox2: Tensor) -> Tensor:
    # there is slight loss of precision in box conversion, so do everythin in one representation
    xy1 = cxcy_to_xyxy(bbox1)
    xy2 = cxcy_to_xyxy(bbox2)

    area1 = (xy1[..., 2:] - xy1[..., :2]).prod(dim=-1)
    area2 = (xy2[..., 2:] - xy2[..., :2]).prod(dim=-1)

    lt = torch.max(xy1[..., :2], xy2[..., :2])
    rb = torch.min(xy1[..., 2:], xy2[..., 2:])
    intersection = (rb - lt).clamp(min=0).prod(dim=-1)
    union = area1 + area2 - intersection
    iou = intersection / union.clamp_min(EPS)
    assert (iou >= 0).all()
    return iou


def compare_aspect_ratios(bbox1: Tensor, bbox2: Tensor) -> Tensor:
    w1, h1 = wh(bbox1).clamp_min(EPS).split(1, dim=-1)
    w2, h2 = wh(bbox2).clamp_min(EPS).split(1, dim=-1)
    result = torch.atan(w2 / h2) - torch.atan(w1 / h1)
    return 4 / pi ** 2 * result.squeeze(-1).pow(2)


def complete_iou_loss(inputs: Tensor, targets: Tensor, reduction: str = "mean", absolute: bool = False) -> Tensor:
    # validation
    check_is_tensor(inputs, "inputs")
    check_is_tensor(targets, "targets")
    check_dimension(inputs, -1, 4, "inputs")
    check_dimension(targets, -1, 4, "targets")
    check_shapes_match(inputs, targets, "inputs", "targets")

    # shortcut for empty input/target
    if not inputs.numel():
        return inputs.new_tensor(0.0)

    with torch.no_grad():
        assert (targets[..., 2:] > 0).all()
        assert (inputs[..., 2:] > 0).all()

    #inputs = inputs.float().clone()
    #targets = targets.float().clone()

    ## convert absolute coordinates to distance from box center if needed
    #if absolute:
    #    inputs = absolute_to_center_delta(inputs)
    #    targets = absolute_to_center_delta(targets)
    #else:
    #    # x1, y1 are negative deltas relative to center
    #    inputs[..., :2] = inputs[..., :2].neg()
    #    targets[..., :2] = targets[..., :2].neg()

    # compute euclidean distance between pred and true box centers
    euclidean_dist_squared = compute_center_distance(inputs, targets).pow(2).sum(dim=-1)

    # compute c, the diagonal length of smallest box enclosing pred and true
    diagonal = compute_enclosing_diagonal(inputs, targets)

    # compute diou
    diou = euclidean_dist_squared #/ diagonal.pow(2)

    # compute vanilla IoU
    iou = compute_iou(inputs, targets)

    # compute v, which measure aspect ratio consistency
    v = compare_aspect_ratios(inputs, targets)

    # compute alpha, the tradeoff parameter
    alpha = v / ((1 - iou) + v).clamp_min(EPS)

    # compute the final ciou loss
    loss = 1 - iou + diou + alpha * v

    #if loss.requires_grad:
    #    print(f"IOU: {iou}")
    #    print(f"DIOU: {diou}")
    #    print(f"V: {v}")

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
