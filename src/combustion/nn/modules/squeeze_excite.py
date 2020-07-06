#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional

import torch.nn as nn
from torch import Tensor

from combustion.nn import HardSigmoid


class _SqueezeExcite(nn.Module):
    def __init__(
        self,
        in_channels: int,
        squeeze_ratio: float,
        out_channels: Optional[int] = None,
        first_activation: nn.Module = nn.ReLU(),
        second_activation: nn.Module = HardSigmoid(),
    ):
        super().__init__()
        self.in_channels = abs(int(in_channels))
        self.squeeze_ratio = abs(float(squeeze_ratio))
        self.out_channels = self.in_channels if out_channels is None else abs(int(out_channels))

        mid_channels = int(max(1, in_channels // squeeze_ratio))

        self.pool = self._get_pool()
        self.linear = nn.Sequential(
            nn.Linear(self.in_channels, mid_channels),
            first_activation,
            nn.Linear(mid_channels, self.out_channels),
            second_activation,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        batch_size, num_channels = inputs.shape[0], inputs.shape[1]
        inputs.ndim - 2

        pooled = self.pool(inputs)
        scale = self.linear(pooled.squeeze().view(batch_size, num_channels)).view_as(pooled)
        return scale

    def _get_pool(self):
        raise NotImplementedError()


class SqueezeExcite1d(_SqueezeExcite):
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
            Defaults to :class:`combustion.nn.HardSwish`.

    Shape
        * Input: :math:`(N, C_i, L)` where :math:`N` is the batch dimension and :math:`C_i` is the channel dimension.
        * Output: :math:`(N, C_o, 1)`.

    .. _Squeeze-and-Excitation Networks:
        https://arxiv.org/abs/1709.01507

    .. _Searching for MobileNetV3:
        https://arxiv.org/abs/1905.02244
    """

    def _get_pool(self):
        return nn.AdaptiveAvgPool1d(output_size=(1,))


class SqueezeExcite2d(_SqueezeExcite):
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
            Defaults to :class:`combustion.nn.HardSwish`.

    Shape
        * Input: :math:`(N, C_i, H, W)` where :math:`N` is the batch dimension and :math:`C_i`
          is the channel dimension.
        * Output: :math:`(N, C_o, 1, 1)`.

    .. _Squeeze-and-Excitation Networks:
        https://arxiv.org/abs/1709.01507

    .. _Searching for MobileNetV3:
        https://arxiv.org/abs/1905.02244 """

    def _get_pool(self):
        return nn.AdaptiveAvgPool2d(output_size=(1, 1))


class SqueezeExcite3d(_SqueezeExcite):
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
            Defaults to :class:`combustion.nn.HardSwish`.

    Shape
        * Input: :math:`(N, C_i, D, H, W)` where :math:`N` is the batch dimension and :math:`C_i`
          is the channel dimension.
        * Output: :math:`(N, C_o, 1, 1, 1)`.

    .. _Squeeze-and-Excitation Networks:
        https://arxiv.org/abs/1709.01507

    .. _Searching for MobileNetV3:
        https://arxiv.org/abs/1905.02244
    """

    def _get_pool(self):
        return nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
