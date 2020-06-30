#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Callable, List, Optional

import torch
import torch.nn as nn
from torch import Tensor


class DefaultConvBlock(nn.Module):
    r"""Default 2D convolution block for BiFPN"""

    def __init__(self, num_channels: int):
        super(DefaultConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1, groups=num_channels, bias=False),
            nn.Conv2d(num_channels, num_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=num_channels, momentum=0.9997, eps=4e-5),
            nn.ReLU(),
        )

    def forward(self, input: Tensor) -> Tensor:
        return self.conv(input)


class _BiFPN_Level(nn.Module):
    __constants__ = ["epsilon"]

    def __init__(
        self,
        num_channels: int,
        conv: Optional[Callable[[int], nn.Module]] = None,
        epsilon: float = 1e-4,
        scale_factor: int = 2,
        weight_2_count: int = 3,
    ):
        super(_BiFPN_Level, self).__init__()

        self.epsilon = float(epsilon)

        if conv is None:
            conv = DefaultConvBlock

        self.conv_up = conv(num_channels)
        self.conv_down = conv(num_channels)

        self.feature_up = nn.Upsample(scale_factor=2, mode="nearest")

        # Conv layers
        self.conv_up = conv(num_channels)
        self.conv_down = conv(num_channels)

        # Feature scaling layers
        self.feature_up = nn.Upsample(scale_factor=scale_factor, mode="nearest")
        self.feature_down = nn.MaxPool2d(kernel_size=scale_factor)

        self.weight_1 = nn.Parameter(torch.ones(2))
        self.weight_2 = nn.Parameter(torch.ones(weight_2_count))

    def forward(
        self, same_level: Tensor, previous_level: Optional[Tensor] = None, next_level: Optional[Tensor] = None
    ) -> Tensor:
        output: Tensor = same_level

        if previous_level is None and next_level is None:
            raise ValueError("previous_level and next_level cannot both be None")

        # input + higher level
        if next_level is not None:
            weight_1 = torch.relu(self.weight_1)
            weight_1 = weight_1 / (torch.sum(weight_1, dim=0) + self.epsilon)

            # weighted combination of current level and higher level
            output = self.conv_up(weight_1[0] * same_level + weight_1[1] * self.feature_up(next_level))

        # input + lower level + last bifpn level (if one exists)
        if previous_level is not None:
            weight_2 = torch.relu(self.weight_2)
            weight_2 = weight_2 / (torch.sum(weight_2, dim=0) + self.epsilon)

            if output is not None:
                # weight_2ed combination of current level, downward fpn output, lower level
                output = self.conv_down(
                    weight_2[0] * same_level + weight_2[1] * output + weight_2[2] * self.feature_down(previous_level)
                )
            # special case for top of pyramid
            else:
                # weighted combination of current level, downward fpn output, lower level
                output = self.conv_down(weight_2[0] * same_level + weight_2[1] * self.feature_down(previous_level))

        return output


class BiFPN(nn.Module):
    r"""A bi-directional feature pyramid network (BiFPN) used in the EfficientDet implementation
    (`EfficientDet Scalable and Efficient Object Detection`_).

    The structure of the block is as follows:

    .. image:: https://miro.medium.com/max/1000/1*qH6d0kBU2cRxOkWUsfgDgg.png
        :width: 300px
        :align: center
        :height: 400px
        :alt: Diagram of BiFPN layer

    Note:

        It is assumed that adjacent levels in the feature pyramid differ in spatial resolution by
        a factor of 2.

    Args:
        num_channels (int):
            The number of channels in each feature pyramid level. All inputs :math:`P_i` should
            have ``num_channels`` channels, and outputs :math:`P_i'` will have ``num_channels`` channels.

        levels (int):
            The number of levels in the feature pyramid.

        conv (callable or torch.nn.Module, optional):
            A function used to override the convolutional layer used. Function must accept one parameter,
            an int equal to ``num_channels``, and return a convolutional layer.
            Default convolutional layer is a separable 2d convolution with batch normalization and relu activation.

        epsilon (float, optional):
            Small value used for numerical stability when normalizing weights.
            Default ``1e-4``.

    Shape:
        - Inputs: Tensor of shape :math:`(N, *C, *H, *W)` where :math:`*C, *H, *W` indicates
          variable channel/height/width at each level of downsapling.
        - Output: Same shape as input.

    .. _EfficientDet Scalable and Efficient Object Detection:
        https://arxiv.org/abs/1911.09070
    """
    __constants__ = ["levels"]

    def __init__(
        self, num_channels: int, levels: int, conv: Optional[Callable[[int], nn.Module]] = None, epsilon: float = 1e-4
    ):
        super(BiFPN, self).__init__()
        if float(epsilon) <= 0.0:
            raise ValueError(f"epsilon must be float > 0, found {epsilon}")
        if int(num_channels) < 1:
            raise ValueError(f"num_channels must be int > 0, found {num_channels}")
        if int(levels) < 1:
            raise ValueError(f"levels must be int > 0, found {levels}")
        if conv is not None and not callable(conv):
            raise ValueError(f"conv must be callable, found {conv}")

        self.levels = levels

        if conv is None:
            conv = DefaultConvBlock

        level_modules = []
        for i in range(levels):
            3 if i > 0 else 2
            level_modules.append(_BiFPN_Level(num_channels, conv, epsilon))
        self.bifpn = nn.ModuleList(level_modules)

    def forward(self, inputs: List[Tensor]) -> List[Tensor]:
        """"""
        outputs: List[Tensor] = []

        for i, layer in enumerate(self.bifpn):
            current_level = inputs[i]
            previous_level = inputs[i - 1] if i > 0 else None
            next_level = inputs[i + 1] if i < len(inputs) - 1 else None
            outputs.append(layer(current_level, previous_level, next_level))

        return outputs
