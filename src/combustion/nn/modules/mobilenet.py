#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional, Tuple, Union

import torch.nn as nn
from torch import Tensor

from .dropconnect import DropConnect
from .squeeze_excite import HardSwish, SqueezeExcite1d, SqueezeExcite2d, SqueezeExcite3d


class _MobileNetMeta(type):
    def __new__(cls, name, bases, dct):
        x = super().__new__(cls, name, bases, dct)
        if "3d" in name:
            x.Conv = nn.Conv3d
            x.BatchNorm = nn.BatchNorm3d
            x.SqueezeExcite = SqueezeExcite3d
        elif "2d" in name:
            x.Conv = nn.Conv2d
            x.BatchNorm = nn.BatchNorm2d
            x.SqueezeExcite = SqueezeExcite2d
        elif "1d" in name:
            x.Conv = nn.Conv1d
            x.BatchNorm = nn.BatchNorm1d
            x.SqueezeExcite = SqueezeExcite1d
        else:
            raise RuntimeError(f"Metaclass: error processing name {cls.__name__}")
        return x


class _MobileNetConvBlockNd(nn.Module):
    def __init__(
        self,
        input_filters: int,
        output_filters: int,
        kernel_size: Union[int, Tuple[int]],
        stride: Union[int, Tuple[int]] = 1,
        bn_momentum: float = 0.1,
        bn_epsilon: float = 1e-5,
        squeeze_excite_ratio: Optional[float] = 1,
        expand_ratio: float = 1,
        use_skipconn: bool = True,
        drop_connect_rate: float = 0.0,
        padding_mode: str = "zeros",
    ):
        super().__init__()
        self._input_filters = int(input_filters)
        self._output_filters = int(output_filters)
        self._kernel_size = kernel_size
        self._stride = stride
        self._bn_momentum = float(bn_momentum)
        self._bn_epsilon = float(bn_epsilon)
        self._se_ratio = abs(float(squeeze_excite_ratio))
        self._expand_ratio = float(expand_ratio)
        self._use_skipconn = bool(use_skipconn)

        padding = (kernel_size - 1) // 2

        # Expansion phase (Inverted Bottleneck)
        in_filter, out_filter = self._input_filters, int(self._input_filters * self._expand_ratio)
        if self._expand_ratio != 1:
            self.expand = nn.Sequential(
                self.Conv(in_filter, out_filter, kernel_size=1, bias=False, padding_mode=padding_mode),
                self.BatchNorm(out_filter, momentum=self._bn_momentum, eps=self._bn_epsilon),
            )
        else:
            self.expand = None

        # Depthwise convolution phase
        depthwise = self.Conv(
            out_filter,
            out_filter,
            self._kernel_size,
            stride=self._stride,
            padding=padding,
            groups=out_filter,
            bias=False,
            padding_mode=padding_mode,
        )
        self.depthwise_conv = nn.Sequential(
            depthwise, self.BatchNorm(out_filter, momentum=self._bn_momentum, eps=self._bn_epsilon)
        )

        # Squeeze and Excitation layer, if desired
        if self._se_ratio is not None:
            self.squeeze_excite = self.SqueezeExcite(out_filter, self._se_ratio)
        else:
            self.squeeze_excite = None

        if drop_connect_rate:
            self.drop_connect = DropConnect(drop_connect_rate)
        else:
            self.drop_connect = None

        # Pointwise convolution phase
        final_out_filter = self._output_filters
        pointwise = self.Conv(out_filter, final_out_filter, kernel_size=1, bias=False, padding_mode=padding_mode,)
        self.pointwise_conv = nn.Sequential(
            pointwise, self.BatchNorm(final_out_filter, momentum=self._bn_momentum, eps=self._bn_epsilon), HardSwish()
        )

    def forward(self, inputs: Tensor) -> Tensor:
        # Expansion and Depthwise Convolution
        x = inputs
        if self.expand is not None:
            x = self.expand(x)

        x = self.depthwise_conv(x)

        # Squeeze and Excitation
        if self.squeeze_excite is not None:
            x_squeezed = self.squeeze_excite(x)
            x = x * x_squeezed

        # Pointwise Convolution
        x = self.pointwise_conv(x)

        # Skip connection and drop connect
        input_filters, output_filters = self._input_filters, self._output_filters
        if self._use_skipconn and self._stride == 1 and input_filters == output_filters:
            if self.drop_connect is not None:
                x = self.drop_connect(x)
            x = x + inputs  # skip connection
        return x


class MobileNetConvBlock1d(_MobileNetConvBlockNd, metaclass=_MobileNetMeta):
    pass


class MobileNetConvBlock2d(_MobileNetConvBlockNd, metaclass=_MobileNetMeta):
    r"""Implementation of the MobileNet inverted bottleneck block as described
    in `Searching for MobileNetV3`_. This implementation includes enhancements from
    MobileNetV3, such as the hard swish activation function (via :class:`combustion.nn.HardSwish`)
    and squeeze/excitation layers  (via :class:`combustion.nn.SqueezeExcite2d`).

    .. image:: ./mobilenet_v3.png
        :width: 600px
        :align: center
        :height: 300px
        :alt: Diagram of MobileNetV3 inverted bottleneck block.

    See :class:`MobileNetConvBlock1d` and :class:`MobileNetConvBlock3d` for 1d / 3d variants.

    Args:
        input_filters (int):
            The number of input channels, :math:`C_i`
            See :class:`torch.nn.Conv2d` for more details.

        output_filters (int):
            Number of output channels, :math:`C_o`
            See :class:`torch.nn.Conv2d` for more details.

        kernel_size (int or tuple of ints):
            Kernel size for the depthwise (spatial) convolutions
            See :class:`torch.nn.Conv2d` for more details.

        stride (int or tuple of ints):
            Stride for the depthwise (spatial) convolutions. See :class:`torch.nn.Conv2d`
            for more details.

        bn_momentum (float):
            Momentum for batch normalization layers. See :class:`torch.nn.BatchNorm2d` for
            more details.

        bn_epsilon (float):
            Epsilon for batch normalization layers. See :class:`torch.nn.BatchNorm2d` for
            more details.

        squeeze_excite_ratio (float):
            Ratio by which channels will be squeezed in the squeeze/excitation layer.
            See :class:`combustion.nn.SqueezeExcite2d` for more details.

        expand_ratio (float):
            Ratio by which channels will be expanded in the inverted bottleneck.

        use_skipconn (bool):
            Whether or not to use skip connections.

        drop_connect_rate (float):
            Drop probability for DropConnect layer. Defaults to ``0.0``, i.e. no
            DropConnect layer will be used.

        padding_mode: str = "zeros"
            Padding mode to use for all non-pointwise convolution layers.
            See :class:`torch.nn.Conv2d` for more details.


    .. _Searching for MobileNetV3:
        https://arxiv.org/abs/1905.02244
    """


class MobileNetConvBlock3d(_MobileNetConvBlockNd, metaclass=_MobileNetMeta):
    r"""3d version of :class:`combustion.nn.MobileNetConvBlock2d`."""
