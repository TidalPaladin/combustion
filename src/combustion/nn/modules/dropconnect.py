#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch import Tensor


class DropConnect(nn.Module):
    r"""Implements DropConnect as defined in `Regularization of Neural Networks using DropConnect`_
    for use with convolutional layers.

    Args:
        ratio (float):
            The ratio of elements to be dropped

    Shape
        * Input: :math:`(N, C, d_1 \dots d_n)` where :math:`d_1 \dots d_n` is any number of
          additional dimensions.
        * Output: Same as input

    .. _Regularization of Neural Networks using DropConnect:
        http://proceedings.mlr.press/v28/wan13.html
    """

    def __init__(self, ratio: float):
        super().__init__()
        self.ratio = 1.0 - abs(float(ratio))
        assert self.ratio >= 0 and self.ratio < 1.0

    def forward(self, x: Tensor) -> Tensor:
        if not self.training:
            return x

        batch_size = x.shape[0]
        mask = self.ratio + torch.rand(batch_size).type_as(x).floor_()
        for i in range(x.ndim - 2):
            mask = mask.unsqueeze(-1)
        return x / self.ratio * mask
