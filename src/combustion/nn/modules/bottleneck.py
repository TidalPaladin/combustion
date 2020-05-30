#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Optional, Tuple, Union

import torch.nn as nn
from torch import Tensor

from .factorized import Conv1d, Conv2d, Conv3d, ConvTranspose1d, ConvTranspose2d, ConvTranspose3d
from .util import double, single, triple


class BottleneckMeta(type):
    def __new__(cls, name, bases, dct):
        x = super().__new__(cls, name, bases, dct)
        if "3d" in name:
            x._tuple = staticmethod(triple)
            if "Factorized" in name:
                x._conv = Conv3d
                x._xconv = ConvTranspose3d
            else:
                x._conv = nn.Conv3d
                x._xconv = nn.ConvTranspose3d
            x._norm = nn.BatchNorm3d
        elif "2d" in name:
            x._tuple = staticmethod(double)
            if "Factorized" in name:
                x._conv = Conv2d
                x._xconv = ConvTranspose2d
            else:
                x._conv = nn.Conv2d
                x._xconv = nn.ConvTranspose2d
            x._norm = nn.BatchNorm2d
        elif "1d" in name:
            x._tuple = staticmethod(single)
            if "Factorized" in name:
                x._conv = Conv1d
                x._xconv = ConvTranspose1d
            else:
                x._conv = nn.Conv1d
                x._xconv = nn.ConvTranspose1d
            x._norm = nn.BatchNorm1d
        x._act = nn.ReLU
        return x


# base class for bottleneck convs
class _BottleneckNd(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, ...]],
        bn_depth: Optional[int] = None,
        bn_spatial: Optional[int] = None,
        stride: Union[int, Tuple[int, ...]] = 1,
        padding: Optional[Union[int, Tuple[int, ...]]] = None,
        dilation: int = 1,
        repeats: int = 1,
        groups: int = 1,
        bias: bool = False,
        padding_mode: str = "zeros",
        checkpoint: bool = False,
    ):
        kernel_size = self._tuple(kernel_size)
        if padding is None:
            padding = tuple([x // 2 for x in kernel_size])
        else:
            padding = self._tuple(padding)
        tuple([x // 2 for x in kernel_size])

        if bn_depth is not None:
            if bn_depth < 1:
                raise ValueError(f"bn_depth {bn_depth} must be >= 1")
            if int(bn_depth) > in_channels:
                raise ValueError(f"bn_depth {bn_depth} must be <= in_channels {in_channels}")
            self.bn_depth = int(bn_depth)
        else:
            self.bn_depth = None

        if bn_spatial is not None:
            bn_spatial = self._tuple(bn_spatial)
            if any([x < 1 for x in bn_spatial]):
                raise ValueError(f"bn_spatial {bn_spatial} must all be >= 1")
            self.bn_spatial = self._tuple(bn_spatial)
        else:
            self.bn_spatial = None

        if int(repeats) < 1:
            raise ValueError(f"repeats must be < 1, got {repeats}")

        super(_BottleneckNd, self).__init__()
        channels = in_channels
        convs = nn.ModuleList([self._act()])

        # NOTE: nonlinearities are omitted for low-dimensional subspaces
        # see section 6 https://arxiv.org/abs/1801.04381

        # enter spatial bottleneck if requested
        if self.bn_spatial is not None:
            # TODO can adapt this based on bn factor?
            kernel = self._tuple(3)
            channels = max(self.bn_spatial) * in_channels
            layer = self._conv(
                in_channels,
                channels,
                kernel,
                stride=self.bn_spatial,
                padding=padding,
                groups=in_channels,
                bias=False,
                padding_mode=padding_mode,
            )
            convs.append(layer)

        # enter channel bottleneck if requested
        if self.bn_depth is not None:
            bn_channels = in_channels // self.bn_depth
            layer = self._conv(channels, bn_channels, 1, bias=False)
            convs.append(layer)
            channels = bn_channels

        # repeated depthwise convolutions in bottleneck
        for i in range(repeats):
            layer = self._conv(
                channels,
                channels,
                kernel_size,
                padding=padding,
                groups=channels,
                bias=False,
                padding_mode=padding_mode,
            )
            convs.append(layer)

        # exit channel bottleneck if requested
        if self.bn_depth is not None:
            out = max(self.bn_spatial) * in_channels if self.bn_spatial is not None else in_channels
            layer = self._conv(channels, out, 1, bias=False)
            convs.append(layer)
            channels = out

        # exit spatial bottleneck if requested
        if self.bn_spatial is not None:
            kernel = self.bn_spatial
            pad = self._tuple(0)
            stri = self.bn_spatial
            layer = self._xconv(
                channels,
                channels,
                kernel,
                stride=stri,
                padding=pad,
                groups=channels,
                bias=False,
                padding_mode=padding_mode,
            )
            convs.append(layer)
            convs.append(self._act())

        # pointwise to final
        layer = self._conv(channels, out_channels, 1, bias=False, padding_mode=padding_mode)
        convs.append(layer)
        self.convs = nn.Sequential(*convs)

    def forward(self, input: Tensor) -> Tensor:
        r""""""
        return self.convs(input)


class Bottleneck3d(_BottleneckNd, metaclass=BottleneckMeta):
    r"""Applies a 3D bottlnecked convolution over an input.
    Bottlnecked convolutions are detailed in the paper
    `MobileNetV2\: Inverted Residuals and Linear Bottlenecks`_ .

    .. note::
        Nonlinearities are omitted for low dimensional subspaces as
        mentioned in section 6 of the paper
        `MobileNetV2\: Inverted Residuals and Linear Bottlenecks`_ ,

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        bn_depth (int): Bottleneck strength in the channel dimension
        bn_spatial (int): Bottleneck strength in the spatial dimensions
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of
            the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel
            elements. Default: 1
        repeats (int): Number of convolutions to perform in the bottlenecked space
        groups (int, optional): Number of blocked connections from input
            channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``
        padding_mode (string, optional): ``'zeros'``, ``'reflect'``, ``'replicate'``
            or ``'circular'``. Default: ``'zeros'``

    .. _MobileNetV2\: Inverted Residuals and Linear Bottlenecks:
        https://arxiv.org/abs/1801.04381
    """


class Bottleneck2d(_BottleneckNd, metaclass=BottleneckMeta):
    r"""Applies a 2D bottlnecked convolution over an input.
    Bottlnecked convolutions are detailed in the paper
    `MobileNetV2\: Inverted Residuals and Linear Bottlenecks`_ .

    .. note::
        Nonlinearities are omitted for low dimensional subspaces as
        mentioned in section 6 of the paper
        `MobileNetV2\: Inverted Residuals and Linear Bottlenecks`_ ,

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        bn_depth (int): Bottleneck strength in the channel dimension
        bn_spatial (int): Bottleneck strength in the spatial dimensions
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of
            the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel
            elements. Default: 1
        repeats (int): Number of convolutions to perform in the bottlenecked space
        groups (int, optional): Number of blocked connections from input
            channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``
        padding_mode (string, optional): ``'zeros'``, ``'reflect'``, ``'replicate'``
            or ``'circular'``. Default: ``'zeros'``

    .. _MobileNetV2\: Inverted Residuals and Linear Bottlenecks:
        https://arxiv.org/abs/1801.04381
    """


class Bottleneck1d(_BottleneckNd, metaclass=BottleneckMeta):
    r"""Applies a 1D bottlnecked convolution over an input.
    Bottlnecked convolutions are detailed in the paper
    `MobileNetV2\: Inverted Residuals and Linear Bottlenecks`_ .

    .. note::
        Nonlinearities are omitted for low dimensional subspaces as
        mentioned in section 6 of the paper
        `MobileNetV2\: Inverted Residuals and Linear Bottlenecks`_ ,

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        bn_depth (int): Bottleneck strength in the channel dimension
        bn_spatial (int): Bottleneck strength in the spatial dimension
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of
            the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel
            elements. Default: 1
        repeats (int): Number of convolutions to perform in the bottlenecked space
        groups (int, optional): Number of blocked connections from input
            channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``
        padding_mode (string, optional): ``'zeros'``, ``'reflect'``, ``'replicate'``
            or ``'circular'``. Default: ``'zeros'``

    .. _MobileNetV2\: Inverted Residuals and Linear Bottlenecks:
        https://arxiv.org/abs/1801.04381
    """


class BottleneckFactorized3d(_BottleneckNd, metaclass=BottleneckMeta):
    r"""Applies a 3D bottlnecked convolution over an input.
    Bottlnecked convolutions are detailed in the paper
    `MobileNetV2\: Inverted Residuals and Linear Bottlenecks`_ . In the
    factorized case, spatial convolutions are performed along each spatial
    dimension separately.

    .. note::
        Nonlinearities are omitted for low dimensional subspaces as
        mentioned in section 6 of the paper
        `MobileNetV2\: Inverted Residuals and Linear Bottlenecks`_ ,

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        bn_depth (int): Bottleneck strength in the channel dimension
        bn_spatial (int): Bottleneck strength in the spatial dimensions
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of
            the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel
            elements. Default: 1
        repeats (int): Number of convolutions to perform in the bottlenecked space
        groups (int, optional): Number of blocked connections from input
            channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``
        padding_mode (string, optional): ``'zeros'``, ``'reflect'``, ``'replicate'``
            or ``'circular'``. Default: ``'zeros'``

    .. _MobileNetV2\: Inverted Residuals and Linear Bottlenecks:
        https://arxiv.org/abs/1801.04381
    """


class BottleneckFactorized2d(_BottleneckNd, metaclass=BottleneckMeta):
    r"""Applies a 2D bottlnecked convolution over an input.
    Bottlnecked convolutions are detailed in the paper
    `MobileNetV2\: Inverted Residuals and Linear Bottlenecks`_ . In the
    factorized case, spatial convolutions are performed along each spatial
    dimension separately.

    .. note::
        Nonlinearities are omitted for low dimensional subspaces as
        mentioned in section 6 of the paper
        `MobileNetV2\: Inverted Residuals and Linear Bottlenecks`_ ,

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        bn_depth (int): Bottleneck strength in the channel dimension
        bn_spatial (int): Bottleneck strength in the spatial dimensions
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of
            the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel
            elements. Default: 1
        repeats (int): Number of convolutions to perform in the bottlenecked space
        groups (int, optional): Number of blocked connections from input
            channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``
        padding_mode (string, optional): ``'zeros'``, ``'reflect'``, ``'replicate'``
            or ``'circular'``.  Default: ``'zeros'``

    .. _MobileNetV2\: Inverted Residuals and Linear Bottlenecks:
        https://arxiv.org/abs/1801.04381
    """


__all__ = [
    "Bottleneck3d",
    "Bottleneck2d",
    "Bottleneck1d",
    "BottleneckFactorized3d",
    "BottleneckFactorized2d",
]
