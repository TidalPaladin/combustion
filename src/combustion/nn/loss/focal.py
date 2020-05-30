#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


_EPSILON = 1e-5


def focal_loss(
    input: Tensor,
    target: Tensor,
    gamma: float,
    pos_weight: Optional[float] = None,
    label_smoothing: Optional[float] = None,
    reduction: str = "mean",
    normalize: bool = False,
):
    r"""Computes the Focal Loss between input and target. See :class:`FocalLoss` for more details

    Args:
        input (torch.Tensor):
            The predicted values on the interval :math:`[0, 1]`.

        target (torch.Tensor):
            The target values on the interval :math:`[0, ``]`.

        gamma (float):
            The focusing parameter :math:`\gamma`. Must be non-negative.

        pos_weight (float, optional):
            The positive weight coefficient :math:`\alpha` to use on
            the positive examples. Must be non-negative.

        label_smoothing (float, optional):
            Float in [0, 1]. When 0, no smoothing occurs. When positive, the binary
            ground truth labels are clamped to :math:`[p, 1-p]`.

        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed.
            Default: ``'mean'``

        normalize (bool, optional):
            If given, output loss will be divided by the number of positive elements
            in ``target``.
    """
    positive_indices = target == 1

    with torch.no_grad():
        target = target.clone().detach()

        if pos_weight is not None:
            alpha = torch.empty_like(input).fill_(1 - pos_weight)
            alpha[positive_indices] = pos_weight
        else:
            alpha = torch.ones_like(input)
        if label_smoothing:
            target.clamp_(label_smoothing, 1.0 - label_smoothing)

    # compute loss
    p = input
    pt = torch.where(target == 1, p, 1 - p)
    ce_loss = F.binary_cross_entropy(input, target, reduction="none")
    loss = alpha * torch.pow(1 - pt, gamma) * ce_loss

    # normalize
    if normalize:
        num_positive_examples = positive_indices.sum().clamp_(min=1)
        loss.div_(num_positive_examples)

    if reduction == "mean":
        loss = loss.mean()
    if reduction == "sum":
        loss = loss.sum()
    return loss


def focal_loss_with_logits(
    input: Tensor,
    target: Tensor,
    gamma: float,
    pos_weight: Optional[float] = None,
    label_smoothing: Optional[float] = None,
    reduction: str = "mean",
    normalize: bool = False,
):
    r"""Computes the Focal Loss between input and target. See :class:`FocalLossWithLogits` for more details

    Args:
        input (torch.Tensor):
            The predicted values.

        target (torch.Tensor):
            The target values.

        gamma (float):
            The focusing parameter :math:`\gamma`. Must be non-negative.

        pos_weight (float, optional):
            The positive weight coefficient :math:`\alpha` to use on
            the positive examples. Must be non-negative.

        label_smoothing (float, optional):
            Float in [0, 1]. When 0, no smoothing occurs. When positive, the binary
            ground truth labels are clamped to :math:`[p, 1-p]`.

        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed.
            Default: ``'mean'``

        normalize (bool, optional):
            If given, output loss will be divided by the number of positive elements
            in ``target``.
    """

    positive_indices = target == 1

    # apply label smoothing, clamping true labels between x, 1-x
    if label_smoothing is not None:
        target = target.clamp(label_smoothing, 1.0 - label_smoothing)

    # loss in paper can be expressed as
    # alpha * (1 - pt) ** gamma * (BCE loss)
    # where pt = p if target == 1 else (1-p)

    # calculate p, pt, and vanilla BCELoss
    # NOTE BCE loss gets logits input, NOT p=sigmoid(input) calulated below
    ce_loss = F.binary_cross_entropy_with_logits(input, target, reduction="none")

    # Therefore one unified form for positive (z = 1) and negative (z = 0)
    # samples is:
    #      (1 - p_t)^r = exp(-r * z * x - r * log(1 + exp(-x))).
    neg_logits = input.neg()

    if gamma != 0:
        focal_term = torch.exp(gamma * (target * neg_logits - neg_logits.exp().log1p()))
        loss = focal_term * ce_loss
    else:
        loss = ce_loss

    if pos_weight is not None:
        loss = torch.where(positive_indices, pos_weight * loss, (1.0 - pos_weight) * loss)

    # normalize
    if normalize:
        num_positive_examples = positive_indices.sum().clamp_(min=1)
        loss.div_(num_positive_examples)

    if reduction == "mean":
        loss = loss.mean()
    if reduction == "sum":
        loss = loss.sum()
    return loss


class _FocalLoss(nn.Module):

    _loss = focal_loss

    def __init__(self, gamma, pos_weight=None, label_smoothing=None, reduction: str = "mean", normalize: bool = False):
        super(_FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = pos_weight
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.normalize = normalize

    @classmethod
    def from_args(cls, args):
        return cls(args.focal_gamma, args.focal_alpha, args.focal_smoothing, args.reduction,)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """forward Calculate smoothed MSE loss between input and target.
        Expects inputs of shape NxCxHxW.

        :param input: The predicted outputs :type input: Tensor :param
        target: The target outputs :type target: Tensor :rtype: Tensor
        """
        loss = self.__class__._loss(
            input, target, self.gamma, self.alpha, self.label_smoothing, self.reduction, self.normalize
        )
        return loss


class FocalLoss(_FocalLoss):
    r"""Creates a criterion that measures the Focal Loss
    between the target and the output. Focal loss is described
    in the paper `Focal Loss For Dense Object Detection`_.

    Args:
        gamma (float):
            The focusing parameter :math:`\gamma`. Must be non-negative.

        pos_weight (float, optional):
            The positive weight coefficient :math:`\alpha` to use on
            the positive examples. Must be non-negative.

        label_smoothing (float, optional):
            Float in [0, 1]. When 0, no smoothing occurs. When positive, the binary
            ground truth labels are clamped to :math:`[p, 1-p]`.

        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed.
            Default: ``'mean'``

        normalize (bool, optional):
            If given, output loss will be divided by the number of positive elements
            in ``target``.

    Shape:
        - Input: :math:`(N, *)` where :math:`*` means, any number of additional
          dimensions
        - Target: :math:`(N, *)`, same shape as the input
        - Output: scalar. If :attr:`reduction` is ``'none'``, then :math:`(N, *)`, same
          shape as input.

    Examples::

        >>> loss = FocalLoss(gamma=1.0, pos_weight=0.8)
        >>> pred = torch.rand(10, 10, requires_grad=True)
        >>> target = torch.rand(10, 10).round()
        >>> output = loss(pred, target)

    .. _Focal Loss For Dense Object Detection:
        https://arxiv.org/abs/1708.02002
    """
    _loss = focal_loss


class FocalLossWithLogits(_FocalLoss):
    r"""Creates a criterion that measures the Focal Loss
    between the target and the output. Focal loss is described
    in the paper `Focal Loss For Dense Object Detection`_. Inputs
    are expected to be logits (i.e. not already scaled to the interval
    :math:`[0, 1]` through a sigmoid or softmax). This computation on
    logits is more numerically stable and efficient for reverse mode
    auto-differentiation and should be preferred for that use case.

    Args:
        gamma (float):
            The focusing parameter :math:`\gamma`. Must be non-negative.

        pos_weight (float, optional):
            The positive weight coefficient :math:`\alpha` to use on
            the positive examples. Must be non-negative.

        label_smoothing (float, optional):
            Float in [0, 1]. When 0, no smoothing occurs. When positive, the binary
            ground truth labels are clamped to :math:`[p, 1-p]`.

        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed.
            Default: ``'mean'``

        normalize (bool, optional):
            If given, output loss will be divided by the number of positive elements
            in ``target``.

    Shape:
        - Input: :math:`(N, *)` where :math:`*` means, any number of additional
          dimensions
        - Target: :math:`(N, *)`, same shape as the input
        - Output: scalar. If :attr:`reduction` is ``'none'``, then :math:`(N, *)`, same
          shape as input.

    Examples::

        >>> loss = FocalLoss(gamma=1.0, pos_weight=0.8)
        >>> pred = torch.rand(10, 10, requires_grad=True)
        >>> target = torch.rand(10, 10).round()
        >>> output = loss(pred, target)

    .. _Focal Loss For Dense Object Detection:
        https://arxiv.org/abs/1708.02002
    """
    _loss = focal_loss_with_logits


__all__ = [
    "focal_loss_with_logits",
    "focal_loss",
    "FocalLoss",
    "FocalLossWithLogits",
]
