#!/usr/bin/env python
# -*- coding: utf-8 -*-

from copy import deepcopy
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from combustion.util import double, single, triple


class _SharedMeta(type):
    def __new__(cls, name, bases, dct):
        x = super().__new__(cls, name, bases, dct)
        if "3d" in name:
            x.Conv = nn.Conv3d
            x.BatchNorm = nn.BatchNorm3d
            x.Tuple = staticmethod(triple)
        elif "2d" in name:
            x.Conv = nn.Conv2d
            x.BatchNorm = nn.BatchNorm2d
            x.Tuple = staticmethod(double)
        elif "1d" in name:
            x.Conv = nn.Conv1d
            x.BatchNorm = nn.BatchNorm1d
            x.Tuple = staticmethod(single)
        else:
            raise RuntimeError(f"Metaclass: error processing name {cls.__name__}")
        return x


class _SharedDecoder(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_convs: int,
        scaled: bool = False,
        strides: Optional[Tuple[int]] = None,
        activation: nn.Module = nn.ReLU(inplace=True),
        final_activation: nn.Module = nn.Identity(),
        num_groups: int = 32,
        gn_epsilon: float = 1e-5,
    ):
        self.scaled = bool(scaled)
        self.strides = strides
        self.num_groups = min(in_channels, num_groups)

        super().__init__()
        for i in range(num_convs):
            # get in/out channels
            is_last_repeat = i == num_convs - 1
            out = out_channels if is_last_repeat else in_channels

            # dw separable conv
            prefix = "final_conv_" if is_last_repeat else f"conv_{i}_"
            dw = self.Conv(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False)
            pw = self.Conv(in_channels, out, 1, bias=is_last_repeat)
            torch.nn.init.normal_(dw.weight, std=0.01)
            torch.nn.init.normal_(pw.weight, std=0.01)

            self.add_module(f"{prefix}dw", dw)
            self.add_module(f"{prefix}pw", pw)

            # bn + act
            if not is_last_repeat:
                gn = nn.GroupNorm(self.num_groups, out, gn_epsilon)
                torch.nn.init.constant_(gn.bias, 0)
                self.add_module(f"gn_{i}", gn)
            else:
                torch.nn.init.constant_(pw.bias, 0)

            self.add_module(f"act_{i}", deepcopy(final_activation if is_last_repeat else activation))

    def forward(self, fpn: Tuple[Tensor]) -> List[Tensor]:
        result: List[Tensor] = []
        assert self.strides is None or len(fpn) == len(self.strides)
        for level_idx, level in enumerate(fpn):
            for module in self:
                level = module(level)

            if self.scaled:
                if self.strides is not None:
                    scale = self.strides[level_idx]
                else:
                    scale = 2 ** level_idx
                level = level * scale
            result.append(level)
        return result


class SharedDecoder2d(_SharedDecoder, metaclass=_SharedMeta):
    r"""Implementation of a FPN decoder / head that is shared across all levels
    of the feature pyramid. The decoder consists of multiple depthwise separable
    convolutional repeats with batch normalization. Outputs can optionally be
    multiplicatively scaled according to the stride at each FPN level.

    .. note::
        Input FPN levels should be sorted in increasing order of stride when
        output ``scaling=True`` is ``strides`` is not specified.

    Args:
        in_channels (int):
            Number of input filters.

        out_channels (int):
            Number of output filters.

        num_convs (int):
            Number of convolutional repeats in the head

        scaled (bool):
            In some cases it is necessary to scale the head outputs by the stride
            of each FPN level. When ``scaled=True``, perform this scaling.

        strides (tuple of ints, optional):
            Strides at each FPN level. Used for applying per-level output scaling when ``scaled=True``.
            When ``scaled=True`` and ``strides`` is not given, assume each FPN level differs by
            a factor of 2.

        activation (nn.Module):
            Activation function for each intermediate repeat in the head.

        final_activation (nn.Module):
            Activation function for the final repeat of the head.

        bn_momentum (float):
            Momentum value for batch norm

        bn_epsilon (float):
            Epsilon value for batch norm

    Returns:
        List of tensors containing the outputs of passing each input FPN level through
        the shared head.

    Shape:
        * ``fpn`` - :math:`(N, C, H_i, W_i)` where :math:`i` is the :math:`i`'th FPN level
        * Output - Same as input
    """


class SharedDecoder1d(_SharedDecoder, metaclass=_SharedMeta):
    pass


class SharedDecoder3d(_SharedDecoder, metaclass=_SharedMeta):
    pass
