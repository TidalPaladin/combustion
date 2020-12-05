#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint as checkpoint_fn

from combustion.util import double, single, triple

from .dropconnect import DropConnect
from .dynamic_pad import DynamicSamePad
from .squeeze_excite import SqueezeExcite1d, SqueezeExcite2d, SqueezeExcite3d


class _MobileNetMeta(type):
    def __new__(cls, name, bases, dct):
        x = super().__new__(cls, name, bases, dct)
        if "3d" in name:
            x.Conv = nn.Conv3d
            x.BatchNorm = nn.BatchNorm3d
            x.SqueezeExcite = SqueezeExcite3d
            x.Tuple = staticmethod(triple)
        elif "2d" in name:
            x.Conv = nn.Conv2d
            x.BatchNorm = nn.BatchNorm2d
            x.SqueezeExcite = SqueezeExcite2d
            x.Tuple = staticmethod(double)
        elif "1d" in name:
            x.Conv = nn.Conv1d
            x.BatchNorm = nn.BatchNorm1d
            x.SqueezeExcite = SqueezeExcite1d
            x.Tuple = staticmethod(single)
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
        dilation: Union[int, Tuple[int]] = 1,
        bn_momentum: float = 0.1,
        bn_epsilon: float = 1e-5,
        activation: nn.Module = nn.Hardswish(),
        squeeze_excite_ratio: Optional[float] = 1,
        expand_ratio: float = 1,
        use_skipconn: bool = True,
        drop_connect_rate: float = 0.0,
        padding_mode: str = "constant",
        global_se: bool = True,
        se_pool_type: Union[str, type] = "avg",
        checkpoint: bool = False,
    ):
        super().__init__()
        kernel_size = self.Tuple(kernel_size)
        stride = self.Tuple(stride)
        dilation = self.Tuple(dilation)

        self._input_filters = int(input_filters)
        self._output_filters = int(output_filters)
        self._kernel_size = kernel_size
        self._stride = stride
        self._dilation = dilation
        self._bn_momentum = float(bn_momentum)
        self._bn_epsilon = float(bn_epsilon)
        self._activation = activation
        self._se_ratio = abs(float(squeeze_excite_ratio))
        self._expand_ratio = float(expand_ratio)
        self._use_skipconn = bool(use_skipconn)
        self._global_se = bool(global_se)
        self._se_pool_type = se_pool_type
        self._checkpoint = bool(checkpoint)

        # Expansion phase (Inverted Bottleneck)
        in_filter, out_filter = self._input_filters, int(self._input_filters * self._expand_ratio)
        if self._expand_ratio != 1:
            self.expand = nn.Sequential(
                self.Conv(in_filter, out_filter, kernel_size=1, bias=False),
                self.BatchNorm(out_filter, momentum=self._bn_momentum, eps=self._bn_epsilon),
                self._activation,
            )
        else:
            self.expand = None

        # Depthwise convolution phase
        depthwise = self.Conv(
            out_filter,
            out_filter,
            self._kernel_size,
            stride=self._stride,
            dilation=self._dilation,
            groups=out_filter,
            bias=False,
        )
        depthwise = DynamicSamePad(depthwise, padding_mode=padding_mode)
        self.depthwise_conv = nn.Sequential(
            depthwise, self.BatchNorm(out_filter, momentum=self._bn_momentum, eps=self._bn_epsilon), self._activation
        )

        # Squeeze and Excitation layer, if desired
        if self._se_ratio is not None:
            ratio = self._se_ratio * self._expand_ratio
            self.squeeze_excite = self.SqueezeExcite(
                out_filter,
                ratio,
                global_pool=self._global_se,
                pool_type=self._se_pool_type,
                first_activation=nn.Hardswish(),
            )
        else:
            self.squeeze_excite = None

        if drop_connect_rate:
            self.drop_connect = DropConnect(drop_connect_rate)
        else:
            self.drop_connect = None

        # Pointwise convolution phase
        final_out_filter = self._output_filters
        pointwise = self.Conv(
            out_filter,
            final_out_filter,
            kernel_size=1,
            bias=False,
        )
        self.pointwise_conv = nn.Sequential(
            pointwise, self.BatchNorm(final_out_filter, momentum=self._bn_momentum, eps=self._bn_epsilon)
        )

    def forward(self, inputs: Tensor) -> Tensor:
        if self._checkpoint and self.training and inputs.requires_grad:
            return self._extract_features_checkpointed(inputs)
        else:
            return self._extract_features(inputs)

    @torch.jit.unused
    def _extract_features_checkpointed(self, inputs: Tensor) -> Tensor:
        return checkpoint_fn(self._extract_features, inputs)

    def _extract_features(self, inputs: Tensor) -> Tensor:
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

    @classmethod
    def from_config(cls, config: "MobileNetBlockConfig") -> Union[nn.Sequential, "_MobileNetConvBlockNd"]:
        r"""Constructs a MobileNetConvBlock using a MobileNetBlockConfig dataclass.

        Args:
            config (:class:`combustion.nn.MobileNetBlockConfig`):
                Configuration for the block to construct
        """
        attrs = [
            "input_filters",
            "output_filters",
            "kernel_size",
            "stride",
            "bn_momentum",
            "bn_epsilon",
            "squeeze_excite_ratio",
            "expand_ratio",
            "use_skipconn",
            "drop_connect_rate",
            "padding_mode",
            "global_se",
            "se_pool_type",
        ]
        kwargs = {attr: getattr(config, attr) for attr in attrs}

        # construct first block
        first_block = cls(**kwargs)

        if config.num_repeats == 1:
            return first_block

        # for multiple repetitions, override filters/stride of blocks 2-N
        kwargs["input_filters"] = config.output_filters
        kwargs["stride"] = 1
        blocks = [first_block] + [cls(**kwargs) for i in range(config.num_repeats - 1)]
        return nn.Sequential(*blocks)


class MobileNetConvBlock1d(_MobileNetConvBlockNd, metaclass=_MobileNetMeta):
    pass


class MobileNetConvBlock2d(_MobileNetConvBlockNd, metaclass=_MobileNetMeta):
    r"""Implementation of the MobileNet inverted bottleneck block as described
    in `Searching for MobileNetV3`_. This implementation includes enhancements from
    MobileNetV3, such as the hard swish activation function (via :class:`torch.nn.Hardswish`)
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

        dilation (int or tuple of ints):
            Dilation of the depthwise (spatial) convolutions. See :class:`torch.nn.Conv2d`
            for more details.

        bn_momentum (float):
            Momentum for batch normalization layers. See :class:`torch.nn.BatchNorm2d` for
            more details.

        bn_epsilon (float):
            Epsilon for batch normalization layers. See :class:`torch.nn.BatchNorm2d` for
            more details.

        activation (:class:`torch.nn.Module`):
            Choice of activation function. Typically this will either be ReLU or Hard Swish
            depending on where the block is located in the network.

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

        padding_mode (str):
            Padding mode to use for all non-pointwise convolution layers.
            See :class:`torch.nn.Conv2d` for more details.

        global_se (bool):
            When ``False``, perform pixel-level squeeze/excitation via a pointwise convolution.
            By default, a channel-level squeeze/excitation is performed via a global pooling step.

        se_pool_type (str or type):
            Type of pooling to use for squeeze/excitation. This only has an effect when ``global_se`` is
            ``True``. See :class:`combustion.nn.SqueezeExcite2d`.

        checkpoint (bool):
            Whether or not to use gradient checkpointing across this block.
            See :func:`torch.utils.checkpoint.checkpoint`.

    .. _Searching for MobileNetV3:
        https://arxiv.org/abs/1905.02244
    """


class MobileNetConvBlock3d(_MobileNetConvBlockNd, metaclass=_MobileNetMeta):
    r"""3d version of :class:`combustion.nn.MobileNetConvBlock2d`."""


@dataclass
class MobileNetBlockConfig:
    r"""Data class that groups parameters for MobileNet inverted bottleneck blocks
    (:class:`MobileNetConvBlock1d`, :class:`MobileNetConvBlock2d`, :class:`MobileNetConvBlock4d`).

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

        dilation (int or tuple of ints):
            Dilation of the depthwise (spatial) convolutions. See :class:`torch.nn.Conv2d`
            for more details.

        bn_momentum (float):
            Momentum for batch normalization layers. See :class:`torch.nn.BatchNorm2d` for
            more details.

        bn_epsilon (float):
            Epsilon for batch normalization layers. See :class:`torch.nn.BatchNorm2d` for
            more details.

        activation (:class:`torch.nn.Module`):
            Choice of activation function. Typically this will either be ReLU or Hard Swish
            depending on where the block is located in the network.

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

        padding_mode (str):
            Padding mode to use for all non-pointwise convolution layers.
            See :class:`torch.nn.Conv2d` for more details.

        global_se (bool):
            When ``False``, perform pixel-level squeeze/excitation via a pointwise convolution.
            By default, a channel-level squeeze/excitation is performed via a global pooling step.

        se_pool_type (str or type):
            Type of pooling to use for squeeze/excitation. This only has an effect when ``global_se`` is
            ``True``. See :class:`combustion.nn.SqueezeExcite2d`.

        checkpoint (bool):
            Whether or not to use gradient checkpointing across this block.
            See :func:`torch.utils.checkpoint.checkpoint`.

        num_repeats (int):
            If given, return a sequence of ``num_repeats`` identical blocks.
    """
    input_filters: int
    output_filters: int
    kernel_size: Union[int, Tuple[int, ...]]
    stride: Union[int, Tuple[int, ...]] = 1
    dilation: Union[int, Tuple[int, ...]] = 1
    bn_momentum: float = 0.1
    bn_epsilon: float = 1e-5
    squeeze_excite_ratio: float = 1.0
    expand_ratio: float = 1.0
    use_skipconn: bool = True
    drop_connect_rate: float = 0.0
    padding_mode: str = "constant"
    global_se: bool = True
    se_pool_type: Union[str, type] = "avg"

    num_repeats: int = 1
    checkpoint: bool = False

    def get_1d_blocks(self, repeated: bool = True) -> Union[MobileNetConvBlock1d, nn.Sequential]:
        return MobileNetConvBlock1d.from_config(self)

    def get_2d_blocks(self, repeated: bool = True) -> Union[MobileNetConvBlock2d, nn.Sequential]:
        return MobileNetConvBlock2d.from_config(self)

    def get_3d_blocks(self, repeated: bool = True) -> Union[MobileNetConvBlock3d, nn.Sequential]:
        return MobileNetConvBlock3d.from_config(self)
