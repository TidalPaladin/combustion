#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import Tensor
import torch.nn as nn
from combustion.nn import MobileNetConvBlock1d, MobileNetConvBlock2d, MobileNetConvBlock3d, MobileNetBlockConfig
from dataclasses import dataclass
from typing import Union, Tuple, List, Optional
from copy import deepcopy
import math


class _EfficientNetMeta(type):
    def __new__(cls, name, bases, dct):
        x = super().__new__(cls, name, bases, dct)
        if "3d" in name:
            x.Conv = nn.Conv3d
            x.BatchNorm = nn.BatchNorm3d
            x._get_blocks = MobileNetBlockConfig.get_3d_blocks
        elif "2d" in name:
            x.Conv = nn.Conv2d
            x.BatchNorm = nn.BatchNorm2d
            x._get_blocks = MobileNetBlockConfig.get_2d_blocks
        elif "1d" in name:
            x.Conv = nn.Conv1d
            x.BatchNorm = nn.BatchNorm1d
            x._get_blocks = MobileNetBlockConfig.get_1d_blocks
        else:
            raise RuntimeError(f"Metaclass: error processing name {cls.__name__}")
        return x

# inspired by https://github.com/lukemelas/EfficientNet-PyTorch/tree/master
class _EfficientNet(nn.Module):
    def __init__(
        self,
        block_configs: List[MobileNetBlockConfig],
        width_coeff: float,
        depth_coeff: float,
        depth_divisor: float = 8.0,
        min_depth: Optional[int] = None,
        stem: Optional[nn.Module] = None,
        head: Optional[nn.Module] = None
    ):
        super().__init__()
        block_configs = deepcopy(block_configs)

        has_non_unit_stride = False
        for config in block_configs:
            # update config according to scale coefficients
            config.input_filters = self.round_filters(config.input_filters, width_coeff, depth_divisor, min_depth)
            config.output_filters = self.round_filters(config.output_filters, width_coeff, depth_divisor, min_depth)
            config.num_repeats = self.round_repeats(depth_coeff, config.num_repeats)
            has_non_unit_stride = has_non_unit_stride or config.stride > 1

        if not has_non_unit_stride:
            import warnings
            warnings.warn(
                "No levels have a non-unit stride. "
                "Unless return_all is True, no extracted features will be returned."
            )

        # Conv stem (default stem used if none given)
        if stem is not None:
            self.stem = stem
        else:
            in_channels = 3
            first_block = next(iter(block_configs))
            output_filters = first_block.input_filters
            bn_momentum = first_block.bn_momentum
            bn_epsilon = first_block.bn_epsilon

            self.stem = nn.Sequential(
                self.Conv(in_channels, output_filters, kernel_size=3, stride=2, bias=False),
                self.BatchNorm(output_filters, momentum=bn_momentum, eps=bn_epsilon),
            )

        # MobileNetV3 convolution blocks
        blocks = []
        for config in block_configs:
            conv_block = self.__class__._get_blocks(config)
            blocks.append(conv_block)
        self.blocks = nn.ModuleList(blocks)

        # Head
        self.head = head

    def extract_features(self, inputs: Tensor, return_all: bool = False) -> List[Tensor]:
        outputs: List[Tensor] = []
        x = self.stem(inputs)
        prev_x = x

        for block in self.blocks:
            x = block(prev_x)

            if return_all or prev_x.shape[-1] > x.shape[-1]:
                outputs.append(x)

            prev_x = x

        return outputs

    def forward(self, inputs: Tensor, use_all_features: bool = False) -> List[Tensor]:
        output = self.extract_features(inputs, use_all_features)
        if self.head is not None:
            output = self.head(output)
            if not isinstance(output, list):
                output = [output,]

        return output

    def round_filters(self, filters, width_coeff, depth_divisor, min_depth):
        if not width_coeff:
            return filters

        filters *= width_coeff
        min_depth = min_depth or depth_divisor 
        new_filters = max(min_depth, int(filters + depth_divisor / 2) // depth_divisor * depth_divisor)

        # prevent rounding by more than 10%
        if new_filters < 0.9 * filters: 
            new_filters += depth_divisor

        return int(new_filters)

    def round_repeats(self, depth_coeff, num_repeats):
        if not depth_coeff:
            return num_repeats
        return int(math.ceil(depth_coeff * num_repeats))



class EfficientNet1d(_EfficientNet, metaclass=_EfficientNetMeta):
    pass


class EfficientNet2d(_EfficientNet, metaclass=_EfficientNetMeta):
    pass


class EfficientNet3d(_EfficientNet, metaclass=_EfficientNetMeta):
    pass
