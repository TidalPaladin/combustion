#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from combustion.util import input


_EPSILON = 1e-5


def focal_loss(input, target, gamma, pos_weight, label_smoothing, reduction="mean"):

    with torch.no_grad():
        target = target.clone().detach()
        alpha = torch.empty_like(input).fill_(1 - pos_weight)
        alpha[target == 1] = pos_weight
        if label_smoothing:
            target.clamp_(label_smoothing, 1.0 - label_smoothing)

    # compute loss
    p = input
    pt = torch.where(target == 1, p, 1 - p)
    ce_loss = F.binary_cross_entropy(input, target, reduction="none")
    loss = alpha * torch.pow(1 - pt, gamma) * ce_loss
    if reduction == "mean":
        loss = loss.mean()
    if reduction == "sum":
        loss = loss.sum()
    return loss


def focal_loss_with_logits(input, target, gamma, pos_weight, label_smoothing, reduction="mean"):

    with torch.no_grad():
        target = target.clone().detach()

        # create alpha tensor with fill:
        # pos_weight if target == 1 else 1 - pos_weight
        alpha = torch.empty_like(input).fill_(1 - pos_weight)
        alpha[target == 1] = pos_weight

        # apply label smoothing, clamping true labels between x, 1-x
        if label_smoothing:
            target.clamp_(label_smoothing, 1.0 - label_smoothing)

    # loss in paper can be expressed as
    # alpha * (1 - pt) ** gamma * (BCE loss)
    # where pt = p if target == 1 else (1-p)

    # calculate p, pt, and vanilla BCELoss
    # NOTE BCE loss gets logits input, NOT p=sigmoid(input) calulated below
    p = torch.sigmoid(input)
    pt = torch.where(target == 1, p, 1 - p)
    ce_loss = F.binary_cross_entropy_with_logits(input, target, reduction="none")
    focal_term = torch.pow(1 - pt, gamma)

    # combine vanilla BCE, focal term
    loss = alpha * focal_term * ce_loss

    if reduction == "mean":
        loss = loss.mean()
    if reduction == "sum":
        loss = loss.sum()
    return loss


class _FocalLoss(nn.Module):
    r"""Creates a criterion that measures the Binary Cross Entropy
    between the target and the output:

    The unreduced (i.e. with :attr:`reduction` set to ``'none'``) loss can be described as:

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = - w_n \left[ y_n \cdot \log x_n + (1 - y_n) \cdot \log (1 - x_n) \right],

    where :math:`N` is the batch size. If :attr:`reduction` is not ``'none'``
    (default ``'mean'``), then

    .. math::
        \ell(x, y) = \begin{cases}
            \operatorname{mean}(L), & \text{if reduction} = \text{'mean';}\\
            \operatorname{sum}(L),  & \text{if reduction} = \text{'sum'.}
        \end{cases}

    This is used for measuring the error of a reconstruction in for example
    an auto-encoder. Note that the targets :math:`y` should be numbers
    between 0 and 1.

    Args:
        gamma : float
            The focusing parameter :math:`\gamma`. Must be non-negative.

        alpha : float, optional
            The coefficient :math:`\alpha` to use on the positive examples.
            Must be non-negative.

        label_smoothing : float, optional
            Float in [0, 1]. When 0, no smoothing occurs. When positive, the binary
            ground truth labels are squeezed toward 0.5, with larger values of
            `label_smoothing` leading to label values closer to 0.5.

        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed.
            Default: ``'mean'``

    Shape:
        - Input: :math:`(N, *)` where :math:`*` means, any number of additional
          dimensions
        - Target: :math:`(N, *)`, same shape as the input
        - Output: scalar. If :attr:`reduction` is ``'none'``, then :math:`(N, *)`, same
          shape as input.

    Examples::

        >>> m = nn.Sigmoid()
        >>> loss = nn.BCELoss()
        >>> input = torch.randn(3, requires_grad=True)
        >>> target = torch.empty(3).random_(2)
        >>> output = loss(m(input), target)
        >>> output.backward()
    """

    _loss = focal_loss

    def __init__(
        self, gamma, alpha, label_smoothing, reduction: str = "mean",
    ):
        super(_FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    @classmethod
    def from_args(cls, args):
        return cls(args.focal_gamma, args.focal_alpha, args.focal_smoothing, args.reduction,)

    @input("input", name=("N", "C", "H", "W"), drop_names=True)
    @input("target", name=("N", "C", "H", "W"), drop_names=True)
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """forward Calculate smoothed MSE loss between input and target.
        Expects inputs of shape NxCxHxW.

        :param input: The predicted outputs :type input: Tensor :param
        target: The target outputs :type target: Tensor :rtype: Tensor
        """
        loss = self.__class__._loss(input, target, self.gamma, self.alpha, self.label_smoothing, self.reduction,)
        return loss


class FocalLoss(_FocalLoss):
    _loss = focal_loss


class FocalLossWithLogits(_FocalLoss):
    _loss = focal_loss_with_logits


__all__ = [
    "focal_loss_with_logits",
    "focal_loss",
    "FocalLoss",
    "FocalLossWithLogits",
]
