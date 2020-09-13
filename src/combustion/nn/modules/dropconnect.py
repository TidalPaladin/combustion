#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch import Tensor


class DropConnect(nn.Module):
    r"""Implements DropConnect as defined in `Regularization of Neural Networks using DropConnect`_
    for use with convolutional layers. This layer is intended to be applied to convolutional layers
    with an accompanying residual connection. DropConnect will zero the output of the main pathway
    with some probability, relying on the skip connection to propagate information.

    Like other dropout layers, DropConnect attempts to force layers to extract features in a way that
    is not dependent on previous layers.

    Args:
        ratio (float):
            The ratio of elements to be dropped

    Example::

        >>> main_path = nn.Conv2d(1, 1, 3, padding=1)
        >>> drop_connect = DropConnect(0.2)
        >>>
        >>> main_output = main_path(inputs)
        >>> # zero drop_connect_output with p=0.2
        >>> drop_connect_output = drop_connect(main_output)
        >>>
        >>> # when drop_connect_output is zeroed, only residual_output remains
        >>> residual_output = inputs
        >>> final_output = drop_connect_output + residual_output

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
        target_shape = (batch_size,) + (1,) * (x.ndim - 1)
        mask = mask.view(target_shape)
        assert mask.ndim == x.ndim
        assert mask.shape[0] == batch_size
        return x / self.ratio * mask
