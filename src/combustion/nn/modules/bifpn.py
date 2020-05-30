#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Callable, Iterable, List, Optional

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

        self.levels = range(int(levels))
        self.epsilon = float(epsilon)

        if conv is None:
            conv = DefaultConvBlock

        # Conv layers
        self.conv_up = nn.ModuleDict({str(x): conv(num_channels) for x in self.levels[:-1]})
        self.conv_down = nn.ModuleDict({str(x): conv(num_channels) for x in reversed(self.levels[1:])})

        # Feature scaling layers
        self.feature_up = nn.ModuleDict({str(x): nn.Upsample(scale_factor=2, mode="nearest") for x in self.levels[:-1]})
        self.feature_down = nn.ModuleDict({str(x): nn.MaxPool2d(kernel_size=2) for x in reversed(self.levels[1:])})

        # Weight
        self.weight_1 = nn.ParameterDict({str(x): nn.Parameter(torch.ones(2)) for x in self.levels[:-1]})
        self.weight_2 = nn.ParameterDict(
            {str(x): nn.Parameter(torch.ones(3 if i > 0 else 2)) for i, x in enumerate(reversed(self.levels[1:]))}
        )

    def forward(self, inputs: Iterable[Tensor]) -> List[Tensor]:
        """"""
        inputs = {x: feature_map for x, feature_map in zip(self.levels, inputs)}
        up_maps, out_maps = {}, {}

        # downward direction
        for level in self.levels[:-1]:
            weight = torch.relu(self.weight_1[str(level)])
            weight = weight / (torch.sum(weight, dim=0) + self.epsilon)
            current_input = inputs[level]
            higher_input = inputs[level + 1]

            # weighted combination of current level and higher level
            output = self.conv_up[str(level)](
                weight[0] * current_input + weight[1] * self.feature_up[str(level)](higher_input)
            )

            # special case for top of pyramid - has no higher level
            if level not in self.weight_2.keys():
                out_maps[level] = output
            else:
                up_maps[level] = output

        # upward direction
        for level in self.levels[1:]:
            weight = torch.relu(self.weight_2[str(level)])
            weight = weight / (torch.sum(weight, dim=0) + self.epsilon)
            current_input = inputs[level]
            lower_input = inputs[level - 1]
            up_map = up_maps[level] if level in up_maps.keys() else None
            assert up_map is None or len(weight) == 3

            if up_map is not None:
                # weighted combination of current level, downward fpn output, lower level
                out_maps[level] = self.conv_down[str(level)](
                    weight[0] * current_input
                    + weight[1] * up_map
                    + weight[2] * self.feature_down[str(level)](lower_input)
                )
            # special case for top of pyramid
            else:
                # weighted combination of current level, downward fpn output, lower level
                out_maps[level] = self.conv_down[str(level)](
                    weight[0] * current_input + weight[1] * self.feature_down[str(level)](lower_input)
                )

        return [out_maps[x] for x in sorted(out_maps.keys())]
