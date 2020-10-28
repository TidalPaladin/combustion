#!/usr/bin/env python
# -*- coding: utf-8 -*-

from copy import deepcopy
from typing import Optional, Union

import torch.nn as nn
from torch import Tensor

from combustion.util import double, single, triple


class _SEMeta(type):
    def __new__(cls, name, bases, dct):
        x = super().__new__(cls, name, bases, dct)
        if "3d" in name:
            x.Conv = nn.Conv3d
            x.BatchNorm = nn.BatchNorm3d
            x.AdaptiveAvgPool = nn.AdaptiveAvgPool3d
            x.AdaptiveMaxPool = nn.AdaptiveMaxPool3d
            x.Tuple = staticmethod(triple)
        elif "2d" in name:
            x.Conv = nn.Conv2d
            x.BatchNorm = nn.BatchNorm2d
            x.AdaptiveAvgPool = nn.AdaptiveAvgPool2d
            x.AdaptiveMaxPool = nn.AdaptiveMaxPool2d
            x.Tuple = staticmethod(double)
        elif "1d" in name:
            x.Conv = nn.Conv1d
            x.BatchNorm = nn.BatchNorm1d
            x.AdaptiveAvgPool = nn.AdaptiveAvgPool1d
            x.AdaptiveMaxPool = nn.AdaptiveMaxPool1d
            x.Tuple = staticmethod(single)
        else:
            raise RuntimeError(f"Metaclass: error processing name {cls.__name__}")
        return x


class _SqueezeExcite(nn.Module):
    def __init__(
        self,
        in_channels: int,
        squeeze_ratio: float,
        out_channels: Optional[int] = None,
        first_activation: nn.Module = nn.ReLU(),
        second_activation: nn.Module = nn.Hardsigmoid(),
        global_pool: bool = True,
        pool_type: Union[str, type] = "avg",
    ):
        super().__init__()
        self.in_channels = abs(int(in_channels))
        self.squeeze_ratio = abs(float(squeeze_ratio))
        self.out_channels = self.in_channels if out_channels is None else abs(int(out_channels))

        mid_channels = int(max(1, in_channels // squeeze_ratio))

        # get pooling type from str or provided module type
        if isinstance(pool_type, str):
            if pool_type == "avg":
                self.pool_type = self.AdaptiveAvgPool
            elif pool_type == "max":
                self.pool_type = self.AdaptiveMaxPool
            else:
                raise ValueError(f"Unknown pool type {pool_type}")
        elif isinstance(pool_type, type):
            self.pool_type = pool_type
        else:
            raise TypeError(f"Expected str or type for pool_type, found {type(pool_type)}")

        # for global attention, squeeze uses global pooling and linear layers
        if global_pool:
            self.pool = self._get_global_pool()
        else:
            self.pool = None

        self.squeeze = nn.Sequential(
            self.Conv(self.in_channels, mid_channels, 1),
            deepcopy(first_activation),
        )
        self.excite = nn.Sequential(
            self.Conv(mid_channels, self.out_channels, 1),
            deepcopy(second_activation),
        )

    def forward(self, inputs: Tensor) -> Tensor:

        # apply global pooling if desired
        if self.pool is not None:
            x = self.pool(inputs)
        else:
            x = inputs

        x = self.squeeze(x)
        x = self.excite(x)
        return x

    def _get_global_pool(self) -> nn.Module:
        target_size = self.Tuple(1)
        return self.pool_type(target_size)


class SqueezeExcite1d(_SqueezeExcite, metaclass=_SEMeta):
    r"""Implements the 1d squeeze and excitation block described in
    `Squeeze-and-Excitation Networks`_, with modifications described in
    `Searching for MobileNetV3`_. Squeeze and excitation layers aid in capturing
    global information embeddings and channel-wise dependencies.

    Channels after the squeeze will be given by

    .. math::
        C_\text{squeeze} = \max\bigg(1, \Big\lfloor\frac{\text{in\_channels}}{\text{squeeze\_ratio}}\Big\rfloor\bigg)

    Args:
        in_channels (int):
            Number of input channels :math:`C_i`.

        squeeze_ratio (float):
            Ratio by which channels will be reduced when squeezing.

        out_channels (optional, int):
            Number of output channels :math:`C_o`. Defaults to ``in_channels``.

        first_activation (:class:`torch.nn.Module`):
            Activation to be applied following the squeeze step.
            Defaults to :class:`torch.nn.ReLU`.

        second_activation (:class:`torch.nn.Module`):
            Activation to be applied following the excitation step.
            Defaults to :class:`torch.nn.Hardsigmoid`.

    Shape
        * Input: :math:`(N, C_i, L)` where :math:`N` is the batch dimension and :math:`C_i` is the channel dimension.
        * Output: :math:`(N, C_o, 1)`.

    .. _Squeeze-and-Excitation Networks:
        https://arxiv.org/abs/1709.01507

    .. _Searching for MobileNetV3:
        https://arxiv.org/abs/1905.02244
    """


class SqueezeExcite2d(_SqueezeExcite, metaclass=_SEMeta):
    r"""Implements the 2d squeeze and excitation block described in
    `Squeeze-and-Excitation Networks`_, with modifications described in
    `Searching for MobileNetV3`_. Squeeze and excitation layers aid in capturing
    global information embeddings and channel-wise dependencies.

    Channels after the squeeze will be given by

    .. math::
        C_\text{squeeze} = \max\bigg(1, \Big\lfloor\frac{\text{in\_channels}}{\text{squeeze\_ratio}}\Big\rfloor\bigg)

    Diagram of the original squeeze/excitation layer

    .. image:: ./squeeze_excite.png
        :width: 400px
        :align: center
        :height: 500px
        :alt: Diagram of MobileNetV3 inverted bottleneck block.

    Args:
        in_channels (int):
            Number of input channels :math:`C_i`.

        squeeze_ratio (float):
            Ratio by which channels will be reduced when squeezing.

        out_channels (optional, int):
            Number of output channels :math:`C_o`. Defaults to ``in_channels``.

        first_activation (:class:`torch.nn.Module`):
            Activation to be applied following the squeeze step.
            Defaults to :class:`torch.nn.ReLU`.

        second_activation (:class:`torch.nn.Module`):
            Activation to be applied following the excitation step.
            Defaults to :class:`torch.nn.Hardsigmoid`.

    Shape
        * Input: :math:`(N, C_i, H, W)` where :math:`N` is the batch dimension and :math:`C_i`
          is the channel dimension.
        * Output: :math:`(N, C_o, 1, 1)`.

    .. _Squeeze-and-Excitation Networks:
        https://arxiv.org/abs/1709.01507

    .. _Searching for MobileNetV3:
        https://arxiv.org/abs/1905.02244"""


class SqueezeExcite3d(_SqueezeExcite, metaclass=_SEMeta):
    r"""Implements the 3d squeeze and excitation block described in
    `Squeeze-and-Excitation Networks`_, with modifications described in
    `Searching for MobileNetV3`_. Squeeze and excitation layers aid in capturing
    global information embeddings and channel-wise dependencies.

    Channels after the squeeze will be given by

    .. math::
        C_\text{squeeze} = \max\bigg(1, \Big\lfloor\frac{\text{in\_channels}}{\text{squeeze\_ratio}}\Big\rfloor\bigg)

    Args:
        in_channels (int):
            Number of input channels :math:`C_i`.

        squeeze_ratio (float):
            Ratio by which channels will be reduced when squeezing.

        out_channels (optional, int):
            Number of output channels :math:`C_o`. Defaults to ``in_channels``.

        first_activation (:class:`torch.nn.Module`):
            Activation to be applied following the squeeze step.
            Defaults to :class:`torch.nn.ReLU`.

        second_activation (:class:`torch.nn.Module`):
            Activation to be applied following the excitation step.
            Defaults to :class:`torch.nn.Hardsigmoid`.

    Shape
        * Input: :math:`(N, C_i, D, H, W)` where :math:`N` is the batch dimension and :math:`C_i`
          is the channel dimension.
        * Output: :math:`(N, C_o, 1, 1, 1)`.

    .. _Squeeze-and-Excitation Networks:
        https://arxiv.org/abs/1709.01507

    .. _Searching for MobileNetV3:
        https://arxiv.org/abs/1905.02244
    """
