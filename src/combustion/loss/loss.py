#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
from argparse import Namespace
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.filters import GaussianBlur2d
from torch import Tensor

from ..data.preprocessing import get_class_weights
from ..util import double
from ..util.pytorch import input
from .focal_loss import FocalLoss, FocalLossWithLogits


def get_criterion(args):
    # type: (Namespace) -> Tuple[nn.Module, nn.Module]
    if args.criterion == "mse":
        train_criterion, eval_criterion = (nn.MSELoss(reduction=args.reduction),) * 2
    elif args.criterion == "bce":
        train_criterion = nn.BCEWithLogitsLoss(reduction=args.reduction)
        eval_criterion = nn.BCELoss(reduction=args.reduction)
    elif args.criterion == "wmse":
        train_criterion, eval_criterion = (WeightedMSELoss.from_args(args),) * 2
    elif args.criterion == "wbce":
        train_criterion = WeightedBCEFromLogitsLoss.from_args(args)
        eval_criterion = WeightedBCELoss.from_args(args)
    elif args.criterion == "sml":
        train_criterion, eval_criterion = (WeightedSoftMarginLoss.from_args(args),) * 2
    elif args.criterion == "focal":
        train_criterion = FocalLossWithLogits.from_args(args)
        eval_criterion = FocalLoss.from_args(args)
    else:
        raise ValueError(f"unknown criterion type {args.criterion}")
    return train_criterion, eval_criterion


class _WeightedLoss(nn.Module):
    _loss = F.binary_cross_entropy

    def __init__(
        self,
        kernel: Optional[Tuple[int, int]] = None,
        sigma: Optional[Tuple[float, float]] = None,
        sparsity: float = 0,
        max_weight: Optional[float] = None,
        reduction: str = "mean",
    ):
        """__init__
        Both `kernel` and `sigma` must be specified, or None. If both are none, only sparsity scaling will be applied.

        :param kernel: The size of the gaussian kernel along x and y
        :type kernel: Optional[Tuple[int, int]]
        :param sigma: The standard deviation of the gaussian kernel along x and y
        :type sigma: Optional[Tuple[float, float]]
        :param sparsity: Bias coefficient lambda for creating sparse outputs.
        :type sparsity: float

        Other args forwarded to torch.nn.MSELoss 
        """
        super(_WeightedLoss, self).__init__()
        self.reduction = reduction
        self.sparsity = sparsity
        self._kernel = double(kernel)
        self._sigma = double(sigma)
        self.max_weight = max_weight
        if kernel is not None and sigma is not None:
            self._gauss = GaussianBlur2d(self._kernel, self._sigma)
        else:
            self._gauss = None

    @classmethod
    def from_args(cls, args):
        return cls(args.smoothing_kernel, args.smoothing_sigma, args.sparsity, args.max_weight, args.reduction)

    @property
    def kernel(self):
        return self._kernel

    @property
    def sigma(self):
        return self._sigma

    @kernel.setter
    def kernel(self, value):
        self._gauss = GaussianBlur2d(value, self.sigma)

    @sigma.setter
    def sigma(self, value):
        self._gauss = GaussianBlur2d(self.kernel, value)

    def get_weights(self, target: torch.Tensor) -> torch.Tensor:
        weights = get_class_weights(target)
        if self.max_weight is not None:
            other = torch.Tensor([self.max_weight]).to(weights)
            weights = torch.min(weights, other)
        weights[target == 0] = 1
        if self._gauss is not None:
            weights = self._gauss(weights)
        return weights

    def get_sparsity_cost(self, input: torch.Tensor) -> torch.Tensor:
        if self.sparsity > 0:
            num_nonzero = len(input[input > 0].nonzero())
            sparsity_cost = (self.sparsity * num_nonzero) / input.numel()
        else:
            sparsity_cost = 0
        return sparsity_cost

    @input("input", name=("N", "C", "H", "W"), drop_names=True)
    @input("target", name=("N", "C", "H", "W"), drop_names=True)
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """forward
        Calculate smoothed MSE loss between input and target. Expects
        inputs of shape NxCxHxW.

        :param input: The predicted outputs
        :type input: Tensor
        :param target: The target outputs
        :type target: Tensor
        :rtype: Tensor
        """
        weights = self.get_weights(target)
        sparsity_cost = self.get_sparsity_cost(input)
        loss = self.__class__._loss(input, target, reduction="none")

        if weights is not None:
            loss = loss.mul_(weights)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss + sparsity_cost


class WeightedBCELoss(_WeightedLoss):
    _loss = F.binary_cross_entropy


class WeightedBCEFromLogitsLoss(_WeightedLoss):
    _loss = F.binary_cross_entropy_with_logits


class WeightedMSELoss(_WeightedLoss):
    _loss = F.mse_loss


class WeightedSoftMarginLoss(_WeightedLoss):
    _loss = F.soft_margin_loss
