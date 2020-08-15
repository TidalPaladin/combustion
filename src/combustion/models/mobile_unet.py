#!/usr/bin/env python
# -*- coding: utf-8 -*-

from copy import deepcopy
from typing import List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from combustion.nn import MatchShapes, MobileNetBlockConfig


class _MobileUnetMeta(type):
    def __new__(cls, name, bases, dct):
        x = super().__new__(cls, name, bases, dct)
        if "3d" in name:
            x.Conv = nn.Conv3d
            x.ConvTranspose = nn.ConvTranspose3d
            x.BatchNorm = nn.BatchNorm3d
            x._get_blocks = MobileNetBlockConfig.get_3d_blocks
        elif "2d" in name:
            x.Conv = nn.Conv2d
            x.ConvTranspose = nn.ConvTranspose2d
            x.BatchNorm = nn.BatchNorm2d
            x._get_blocks = MobileNetBlockConfig.get_2d_blocks
        elif "1d" in name:
            x.Conv = nn.Conv1d
            x.ConvTranspose = nn.ConvTranspose1d
            x.BatchNorm = nn.BatchNorm1d
            x._get_blocks = MobileNetBlockConfig.get_1d_blocks
        else:
            raise RuntimeError(f"Metaclass: error processing name {cls.__name__}")
        return x


class _MobileUnet(nn.Module):
    def __init__(
        self,
        down_configs: List[MobileNetBlockConfig],
        up_configs: Optional[List[MobileNetBlockConfig]] = None,
        stem: Optional[nn.Module] = None,
        head: Optional[nn.Module] = None,
    ):
        super().__init__()
        down_configs = deepcopy(down_configs)
        if up_configs is not None:
            up_configs = deepcopy(up_configs)
        else:
            up_configs = []
            for config in reversed(down_configs):
                reversed_config = deepcopy(config)
                reversed_config.input_filters = config.output_filters
                reversed_config.output_filters = config.input_filters
                up_configs.append(reversed_config)

        # stem / head if provided
        self.stem = stem if stem is not None else nn.Identity()
        self.head = head if head is not None else nn.Identity()

        # MobileNetV3 convolution blocks for downsampling
        blocks = []
        for config in down_configs:
            conv_block = self.__class__._get_blocks(config)
            blocks.append(conv_block)
        self.down_blocks = nn.ModuleList(blocks)

        # MobileNetV3 convolution blocks for upsampling
        blocks = []
        for i, config in enumerate(up_configs):
            stride = config.stride

            in_channels = config.input_filters
            out_channels = config.output_filters
            config.output_filters = in_channels

            # special case for first up level with no skip conn
            if i != 0:
                config.input_filters *= 2

            config.stride = 1
            conv_block = nn.Sequential(
                self.__class__._get_blocks(config),
                self.ConvTranspose(in_channels, out_channels, kernel_size=2, stride=stride),
            )
            blocks.append(conv_block)
        self.up_blocks = nn.ModuleList(blocks)

        self.match_shapes = MatchShapes(strategy="crop")

    def forward(self, inputs: Tensor) -> Tensor:
        _ = self.stem(inputs)

        # downsampling levels
        skip_conns: List[Tensor] = [_]
        for down_level in self.down_blocks:
            _ = down_level(_)
            skip_conns.append(_)
        del skip_conns[-1]

        for i, up_level in enumerate(self.up_blocks):
            # upsample
            _ = up_level(_)

            # match skip conn shape and cat
            skip_conn = skip_conns[-(i + 1)]
            spatial_shape = skip_conn.shape[2:]
            _ = self.match_shapes([_], spatial_shape)[0]
            _ = torch.cat([_, skip_conn], dim=1)

        _ = self.head(_)
        return _

    @classmethod
    def from_identical_blocks(
        cls,
        block: MobileNetBlockConfig,
        in_channels: int,
        levels: List[int],
        stem: Optional[nn.Module] = None,
        head: Optional[nn.Module] = None,
    ) -> "_MobileUnet":
        down_blocks: List[MobileNetBlockConfig] = []
        for i in levels:
            new_block = deepcopy(block)
            new_block.num_repeats = i
            new_block.input_filters = in_channels
            new_block.output_filters = in_channels * 2
            new_block.stride = 2
            down_blocks.append(new_block)
            in_channels *= 2

        return cls(down_blocks, stem=stem, head=head)


class MobileUnet1d(_MobileUnet, metaclass=_MobileUnetMeta):
    pass


class MobileUnet2d(_MobileUnet, metaclass=_MobileUnetMeta):
    r"""Modified implementation of U-Net as described in the `U-Net paper`_. This implementation
    uses MobileNetV3 inverted bottleneck blocks (from `Searching for MobileNetV3`_) as the fundamental
    building block of each convolutional layer. Automatic padding/cropping is used to ensure operation
    with an input of arbitrary spatial shape.

    A general U-Net architecture is as follows:

    .. image:: unet.png
        :width: 800px
        :align: center
        :height: 500px
        :alt: Diagram of BiFPN layer

    Args:
        down_configs (list of :class:`combustion.nn.MobileNetBlockConfig`)
            Configs for each of the :class:`combustion.nn.MobileNetConvBlock2d` blocks
            used in the downsampling portion of the model.

        up_configs (optional, list of :class:`combustion.nn.MobileNetBlockConfig`)
            Configs for each of the :class:`combustion.nn.MobileNetConvBlock2d` blocks
            used in the upsampling portion of the model. By default, the reverse ``down_configs``
            is used with as-needed modifications.

        stem (optional, :class:`torch.nn.Module`):
            An stem/tail layer.

        head (optional, :class:`torch.nn.Module`):
            An optional head layer

    Shapes
        * Input: :math:`(N, C, H, W)`
        * Output: List of tensors of shape :math:`(N, C, H', W')`, where height and width vary
          depending on the amount of downsampling for that feature map.

    .. _U-Net paper:
        https://arxiv.org/abs/1505.04597

    .. _Searching for MobileNetV3:
        https://arxiv.org/abs/1905.02244
    """


class MobileUnet3d(_MobileUnet, metaclass=_MobileUnetMeta):
    pass
