#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Function


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
    #
    # Becuase logits are unbounded, log(1 + exp(-x)) must be computed using
    # torch.logaddexp()
    neg_logits = input.neg().float()

    if gamma != 0:
        _ = torch.tensor([0.0], device=neg_logits.device).type_as(neg_logits)
        _ = gamma * (target.floor() * neg_logits - torch.logaddexp(neg_logits, _))
        focal_term = torch.exp(_)
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
        return cls(
            args.focal_gamma,
            args.focal_alpha,
            args.focal_smoothing,
            args.reduction,
        )

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
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


class Log1mExp(Function):
    @staticmethod
    def forward(ctx, x):
        case1 = torch.log(torch.expm1(x).neg())
        case2 = torch.log1p(x.exp().neg())
        # return torch.where(x > -0.693147, case1, case2)
        result = torch.where(x > -0.693147, case1, case2)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        # d/dx log(1-e^x) = -e^x / (1 - e^x)
        ...


@torch.jit.script
def log1mexp(x: Tensor) -> Tensor:
    case1 = torch.log(torch.expm1(x).neg())
    case2 = torch.log1p(x.exp().neg())
    # return torch.where(x > -0.693147, case1, case2)
    return torch.where(x > -0.693147, case1, case2)


def log1m(x: Tensor) -> Tensor:
    return torch.log1p(x.neg())


def categorical_focal_loss(
    input: Tensor,
    target: Tensor,
    gamma: float,
    pos_weight: Optional[float] = None,
    label_smoothing: Optional[float] = None,
    reduction: str = "mean",
    normalize: bool = False,
):
    r"""Computes the categorical Focal Loss between input and target. This is a multi-class
    loss function.

    .. warning::
        This method is experimental

    Args:
        input (torch.Tensor):
            The predicted values.

        target (torch.Tensor):
            The target values.

        gamma (float):
            The focusing parameter :math:`\gamma`. Must be non-negative.

        pos_weight (float or iterable of floats, optional):
            The positive weight coefficient :math:`\alpha` to use on
            the positive examples. Must be in range :math:`[0, 1]`.
            If an iterable is given, length of iterable should match expected
            number of classes

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
    num_classes = input.shape[1]
    positive_indices = target == 1

    # apply label smoothing, clamping true labels between x, 1-x
    if label_smoothing is not None:
        target = target.clamp(label_smoothing, 1.0 - label_smoothing)

    # loss in paper can be expressed as
    # alpha * (1 - pt) ** gamma * (CE loss)
    # where pt = p if target == 1 else (1-p)

    # calculate p, pt, and vanilla CELoss
    # NOTE CE loss gets logits input
    # NOTE for some reason, target.clone() must be used or inplace op errors arise
    ce_loss = F.cross_entropy(input.float(), target.clone(), reduction="none")

    # Let S = sum_{i=1}^n x_j (denominator of softmax), r = gamma, z = 1 or 0 positive example indicator
    #
    # For positive example (z=1) case:
    #
    # (1 - p_t)^r = (1 - e^x_i / S)^r
    #   = exp(r * log(1 - e^x_i / S))
    #   = exp(r * log(1 - e^x_i / e^(log(S))))
    #   = exp(r * [log(sum_{j != i} e^x_j) - log(S)])
    #
    # For negative example (z=0) case:
    #
    # (1 - p_t)^r = (1 - (1 - e^x_i / S))^r
    #   = (e^x_i / S)^r
    #   = exp(r * log(e^x_i / S))
    #   = exp(r * log(e^x_i) - r * log(S))))
    #   = exp(r * [x_i - r * log(S)))])
    z = F.one_hot(target, num_classes=num_classes).type_as(input).float()

    if pos_weight is not None:
        raise NotImplementedError("positive example weighting not yet implemented")
        pos_weight = torch.as_tensor(pos_weight, device=ce_loss.device, dtype=ce_loss.dtype)
        ce_loss = torch.where(positive_indices, pos_weight * ce_loss, (1.0 - pos_weight) * ce_loss)

    input = input.movedim(1, -1).view(-1, num_classes)
    positive_indices = positive_indices.view(-1)
    z = z.view(-1, num_classes)

    if gamma != 0:
        logS = torch.logsumexp(input, dim=-1, keepdim=True)
        log1mexp(input - logS)

        # get exp sum of logits at z=0 indices
        non_pos_logits = input[~z.bool()].view(-1, num_classes - 1)
        neg = torch.exp(gamma * F.log_softmax(input, dim=1))
        pos = torch.exp(gamma * (torch.logsumexp(non_pos_logits, dim=1, keepdim=True) - logS)).expand_as(neg)
        focal_term = torch.where(z == 1, pos, neg)
        loss = torch.sum(focal_term * (ce_loss.view(-1, 1) / num_classes), dim=1)
    else:
        loss = ce_loss

    loss = loss.view_as(target)

    if reduction == "mean":
        loss = loss.mean()
    if reduction == "sum":
        loss = loss.sum()
    return loss


class CategoricalFocalLoss(nn.Module):
    r"""Computes the Focal Loss between input and target. This is a multi-class loss function, and is
    numerically stabilized.

    .. warning::
        This method is experimental

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
    """

    def __init__(self, gamma, pos_weight=None, label_smoothing=None, reduction: str = "mean", normalize: bool = False):
        super(CategoricalFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = pos_weight
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.normalize = normalize

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        loss = categorical_focal_loss(
            input, target, self.gamma, self.alpha, self.label_smoothing, self.reduction, self.normalize
        )
        return loss
