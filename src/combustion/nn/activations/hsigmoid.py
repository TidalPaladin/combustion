#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# implementation inspired by
# https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/utils.py


def hard_sigmoid(inputs: Tensor, inplace: bool = True) -> Tensor:
    r"""The hard sigmoid activation function, defined as

    .. math::
        f(x) = \frac{\text{ReLU6}(x + 3)}{6}

    .. warning::
        Deprecated in favor of :func:`torch.nn.functional.hardsigmoid`

    Hard sigmoid is a computationally efficient approximation to the sigmoid activation
    and is more suitable for quantization.

    Args:

        inputs (Tensor):
            The input tensor

        inplace (bool, optional):
            Whether or not to perform the operation in place.
    """
    if inplace:
        return F.relu6(inputs.add_(3), inplace=True).div_(6)
    else:
        return F.relu6(inputs + 3).div(6)


class HardSigmoid(nn.Module):
    r"""The hard sigmoid activation function, defined as

    .. math::
        f(x) = \frac{\text{ReLU6}(x + 3)}{6}

    .. warning::
        Deprecated in favor of :func:`torch.nn.HardSigmoid`

    Hard sigmoid is a computationally efficient approximation to the sigmoid activation
    and is more suitable for quantization.

    .. image:: ./hsigmoid.png
        :width: 400px
        :align: center
        :height: 200px
        :alt: Comparison of hard sigmoid and sigmoid activations.

    Args:

        inplace (bool, optional):
            Whether or not to perform the operation in place.
    """

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = bool(inplace)

    def forward(self, inputs: Tensor) -> Tensor:
        return hard_sigmoid(inputs, self.inplace)
