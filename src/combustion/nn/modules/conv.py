#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
from typing import Tuple, Union

import torch
import torch.nn as nn

from combustion.util import double, triple

from .bottleneck import Bottleneck2d, Bottleneck3d, BottleneckFactorized2d, BottleneckFactorized3d
from .factorized import Conv2d, Conv3d, ConvTranspose2d, ConvTranspose3d


# type hints
Kernel2D = Union[Tuple[int, int], int]
Kernel3D = Union[Tuple[int, int, int], int]
Pad2D = Union[Tuple[int, int], int]
Pad3D = Union[Tuple[int, int, int], int]
Head = Union[bool, nn.Module]


class _RepeatFinal(nn.Module):
    def __init__(self, repeated, final, num_repeats):
        super(_RepeatFinal, self).__init__()
        convs = nn.ModuleList([copy.deepcopy(repeated) for i in range(num_repeats)])
        self.repeated = nn.Sequential(*convs)
        self.final = final

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        pre_final = self.repeated(input)
        final = self.final(pre_final)
        return final, pre_final


class DownSample3d(_RepeatFinal):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        factorized=False,
        bn_depth=None,
        bn_spatial=None,
        repeats=1,
        bn_repeats=1,
        stride=2,
        padding=0,
        groups=1,
        bias=False,
        padding_mode="zeros",
        checkpoint=False,
    ):
        kernel_size = triple(kernel_size)
        same_pad = tuple([x // 2 for x in kernel_size])
        if bn_depth is not None or bn_spatial is not None:
            if factorized:

                repeated = BottleneckFactorized3d(
                    in_channels,
                    in_channels,
                    kernel_size,
                    bn_depth,
                    bn_spatial,
                    1,
                    same_pad,
                    1,
                    bn_repeats,
                    groups,
                    bias,
                    padding_mode,
                    checkpoint,
                )
            else:
                repeated = Bottleneck3d(
                    in_channels,
                    in_channels,
                    kernel_size,
                    bn_depth,
                    bn_spatial,
                    1,
                    same_pad,
                    1,
                    bn_repeats,
                    groups,
                    bias,
                    padding_mode,
                    checkpoint,
                )
        else:
            if factorized:
                repeated = Conv3d(
                    in_channels,
                    in_channels,
                    kernel_size,
                    1,
                    same_pad,
                    1,
                    groups,
                    bias,
                    padding_mode,
                )
            else:
                repeated = nn.Conv3d(
                    in_channels,
                    in_channels,
                    kernel_size,
                    1,
                    same_pad,
                    1,
                    groups,
                    bias,
                    padding_mode,
                )

        if factorized:
            final = Conv3d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                1,
                groups,
                bias,
                padding_mode,
            )
        else:
            final = nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                1,
                groups,
                bias,
                padding_mode,
            )

        super(DownSample3d, self).__init__(repeated, final, repeats)


class UpSample3d(_RepeatFinal):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        up_kernel_size,
        factorized=False,
        bn_depth=None,
        bn_spatial=None,
        repeats=1,
        bn_repeats=1,
        stride=2,
        padding=0,
        groups=1,
        bias=False,
        padding_mode="zeros",
        checkpoint=False,
    ):
        kernel_size = triple(kernel_size)
        same_pad = tuple([x // 2 for x in kernel_size])
        if bn_depth is not None or bn_spatial is not None:
            if factorized:
                repeated = BottleneckFactorized3d(
                    in_channels,
                    in_channels,
                    kernel_size,
                    bn_depth,
                    bn_spatial,
                    1,
                    same_pad,
                    1,
                    bn_repeats,
                    groups,
                    bias,
                    padding_mode,
                    checkpoint,
                )
            else:
                repeated = Bottleneck3d(
                    in_channels,
                    in_channels,
                    kernel_size,
                    bn_depth,
                    bn_spatial,
                    1,
                    same_pad,
                    1,
                    bn_repeats,
                    groups,
                    bias,
                    padding_mode,
                    checkpoint,
                )
        else:
            if factorized:
                repeated = Conv3d(
                    in_channels,
                    in_channels,
                    kernel_size,
                    1,
                    same_pad,
                    1,
                    groups,
                    bias,
                    padding_mode,
                )
            else:
                repeated = nn.Conv3d(
                    in_channels,
                    in_channels,
                    kernel_size,
                    1,
                    same_pad,
                    1,
                    groups,
                    bias,
                    padding_mode,
                )

        if factorized:
            final = ConvTranspose3d(
                in_channels,
                out_channels,
                up_kernel_size,
                stride,
                padding,
                0,
                groups,
                bias,
                1,
                padding_mode,
            )
        else:
            final = nn.ConvTranspose3d(
                in_channels,
                out_channels,
                up_kernel_size,
                stride,
                padding,
                0,
                groups,
                bias,
                1,
                padding_mode,
            )

        super(UpSample3d, self).__init__(repeated, final, repeats)


class UpSample2d(_RepeatFinal):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        up_kernel_size,
        factorized=False,
        bn_depth=None,
        bn_spatial=None,
        repeats=1,
        bn_repeats=1,
        stride=2,
        padding=0,
        groups=1,
        bias=False,
        padding_mode="zeros",
        checkpoint=False,
    ):
        kernel_size = double(kernel_size)
        same_pad = tuple([x // 2 for x in kernel_size])
        if bn_depth is not None or bn_spatial is not None:
            if factorized:
                repeated = BottleneckFactorized2d(
                    in_channels,
                    in_channels,
                    kernel_size,
                    bn_depth,
                    bn_spatial,
                    1,
                    same_pad,
                    1,
                    bn_repeats,
                    groups,
                    bias,
                    padding_mode,
                    checkpoint,
                )
            else:
                repeated = Bottleneck2d(
                    in_channels,
                    in_channels,
                    kernel_size,
                    bn_depth,
                    bn_spatial,
                    1,
                    same_pad,
                    1,
                    bn_repeats,
                    groups,
                    bias,
                    padding_mode,
                    checkpoint,
                )
        else:
            if factorized:
                repeated = Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size,
                    1,
                    same_pad,
                    1,
                    groups,
                    bias,
                    padding_mode,
                )
            else:
                repeated = nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size,
                    1,
                    same_pad,
                    1,
                    groups,
                    bias,
                    padding_mode,
                )

        if factorized:
            final = ConvTranspose2d(
                in_channels,
                out_channels,
                up_kernel_size,
                stride,
                padding,
                0,
                groups,
                bias,
                1,
                padding_mode,
            )
        else:
            final = nn.ConvTranspose2d(
                in_channels,
                out_channels,
                up_kernel_size,
                stride,
                padding,
                0,
                groups,
                bias,
                1,
                padding_mode,
            )

        super(UpSample2d, self).__init__(repeated, final, repeats)
