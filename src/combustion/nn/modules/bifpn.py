#!/usr/bin/env python
# -*- coding: utf-8 -*-

from copy import deepcopy
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint as checkpoint_fn

from combustion.util import double, single, triple

from ..activations import HardSwish


class _BiFPNMeta(type):
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


class _BiFPN_Level(nn.Module):
    __constants__ = ["epsilon"]

    def __init__(
        self,
        num_channels: int,
        upsample_mode: str,
        conv: Callable[[int], nn.Module],
        epsilon: float = 1e-4,
        scale_factor: Union[int, Tuple[int, ...]] = 2,
        weight_2_count: int = 3,
    ):
        super(_BiFPN_Level, self).__init__()

        self.epsilon = float(epsilon)
        self.upsample_mode = str(upsample_mode)

        # Conv layers
        self.conv_up = conv(num_channels)
        self.conv_down = conv(num_channels)

        self.weight_1 = nn.Parameter(torch.ones(2))
        self.weight_2 = nn.Parameter(torch.ones(weight_2_count))

    def forward(
        self, same_level: Tensor, previous_level: Optional[Tensor] = None, next_level: Optional[Tensor] = None
    ) -> Tensor:
        output: Tensor = same_level

        if previous_level is None and next_level is None:
            raise ValueError("previous_level and next_level cannot both be None")

        target_shape = same_level.shape[2:]

        # input + higher level
        if next_level is not None:
            weight_1 = torch.relu(self.weight_1)
            weight_1 = weight_1 / (torch.sum(weight_1, dim=0) + self.epsilon)

            # weighted combination of current level and higher level
            next_level = F.interpolate(next_level, target_shape, mode=self.upsample_mode)
            output = self.conv_up(weight_1[0] * same_level + weight_1[1] * next_level)

        # input + lower level + last bifpn level (if one exists)
        if previous_level is not None:
            weight_2 = torch.relu(self.weight_2)
            weight_2 = weight_2 / (torch.sum(weight_2, dim=0) + self.epsilon)
            previous_level = self.pool(previous_level, target_shape)

            if output is not None:
                # weight_2ed combination of current level, downward fpn output, lower level
                output = self.conv_down(weight_2[0] * same_level + weight_2[1] * output + weight_2[2] * previous_level)
            # special case for top of pyramid
            else:
                # weighted combination of current level, downward fpn output, lower level
                output = self.conv_down(weight_2[0] * same_level + weight_2[1] * previous_level)

        return output

    def pool(self, inputs: Tensor, target_shape: List[int]) -> Tensor:
        if len(target_shape) == 2:
            return F.adaptive_max_pool2d(inputs, target_shape)
        elif len(target_shape) == 3:
            return F.adaptive_max_pool3d(inputs, target_shape)
        elif len(target_shape) == 1:
            return F.adaptive_max_pool1d(inputs, target_shape)
        else:
            raise RuntimeError(f"Invalid target_shape: {target_shape}")


class _BiFPN(nn.Module):
    __constants__ = ["levels"]

    def __init__(
        self,
        num_channels: int,
        levels: int,
        kernel_size: Union[int, Tuple[int, ...]] = 3,
        stride: Union[int, Tuple[int, ...]] = 2,
        epsilon: float = 1e-4,
        bn_momentum: float = 0.001,
        bn_epsilon: float = 4e-5,
        activation: nn.Module = HardSwish(),
        upsample_mode: str = "nearest",
        checkpoint: bool = False,
    ):
        super().__init__()
        if float(epsilon) <= 0.0:
            raise ValueError(f"epsilon must be float > 0, found {epsilon}")
        if int(num_channels) < 1:
            raise ValueError(f"num_channels must be int > 0, found {num_channels}")
        if int(levels) < 1:
            raise ValueError(f"levels must be int > 0, found {levels}")

        upsample_mode = str(upsample_mode)

        self.levels = levels
        kernel_size = self.Tuple(kernel_size)
        stride = self.Tuple(stride)
        padding = tuple([(kernel - 1) // 2 for kernel in kernel_size])
        self.checkpoint = bool(checkpoint)

        def conv(num_channels):
            return nn.Sequential(
                deepcopy(activation),
                self.Conv(num_channels, num_channels, kernel_size, padding=padding, groups=num_channels, bias=False),
                self.Conv(num_channels, num_channels, kernel_size=1, bias=False),
                self.BatchNorm(num_features=num_channels, momentum=bn_momentum, eps=bn_epsilon),
            )

        level_modules = []
        for i in range(levels):
            weight_2_count = 3 if i > 0 else 2
            level_modules.append(_BiFPN_Level(num_channels, upsample_mode, conv, epsilon, stride, weight_2_count))
        self.bifpn = nn.ModuleList(level_modules)

    def forward(self, inputs: List[Tensor]) -> List[Tensor]:
        """"""
        if self.checkpoint and self.training and all([x.requires_grad for x in inputs]):
            return self._extract_features_checkpointed(inputs)
        else:
            return self._extract_features(inputs)

    @torch.jit.unused
    def _extract_features_checkpointed(self, inputs: List[Tensor]) -> List[Tensor]:
        outputs: List[Tensor] = []

        for i, layer in enumerate(self.bifpn):
            current_level = inputs[i]
            previous_level = inputs[i - 1] if i > 0 else None
            next_level = inputs[i + 1] if i < len(inputs) - 1 else None
            output = checkpoint_fn(layer, current_level, previous_level, next_level)
            outputs.append(output)

        return outputs

    def _extract_features(self, inputs: List[Tensor]) -> List[Tensor]:
        outputs: List[Tensor] = []

        for i, layer in enumerate(self.bifpn):
            current_level = inputs[i]
            previous_level = inputs[i - 1] if i > 0 else None
            next_level = inputs[i + 1] if i < len(inputs) - 1 else None
            outputs.append(layer(current_level, previous_level, next_level))

        return outputs


class BiFPN1d(_BiFPN, metaclass=_BiFPNMeta):
    pass


class BiFPN2d(_BiFPN, metaclass=_BiFPNMeta):
    r"""A bi-directional feature pyramid network (BiFPN) used in the EfficientDet implementation
    (`EfficientDet Scalable and Efficient Object Detection`_). The bi-directional FPN mixes features
    at different resolution, while also capturing (via learnable weights) that features at different
    resolutions can contribute unequally to the desired output.

    Weights controlling the contribution of each FPN level are normalized using fast normalized fusion,
    which the authors note is more efficient than a softmax based fusion. It is ensured that for all
    weights, :math:`w_i > 0` by applying ReLU to each weight.

    The weight normalization is as follows

    .. math::
        O = \sum_{i}\frac{w_i}{\epsilon + \sum_{j} w_j} \cdot I_i

    The structure of the block is as follows:

    .. image:: https://miro.medium.com/max/1000/1*qH6d0kBU2cRxOkWUsfgDgg.png
        :width: 300px
        :align: center
        :height: 400px
        :alt: Diagram of BiFPN layer

    .. note::
        This implementation will automatically match spatial shapes between BiFPN levels. Adjacent levels
        are upsampled / downsampled by a factor of 2 and padded/cropped to ensure shapes match.

    Args:

        num_channels (int):
            The number of channels in each feature pyramid level. All inputs :math:`P_i` should
            have ``num_channels`` channels, and outputs :math:`P_i'` will have ``num_channels`` channels.

        levels (int):
            The number of levels in the feature pyramid. Must have ``levels > 1``.

        kernel_size (int or tuple of ints):
            Choice of kernel size

        stride (int or tuple of ints):
            Controls the scaling used to upsample/downsample adjacent levels in the BiFPN. This stride
            is passed to :class:`torch.nn.MaxPool2d` and :class:`torch.nn.Upsample`.

        epsilon (float, optional):
            Small value used for numerical stability when normalizing weights via fast normalized fusion.
            Default ``1e-4``.

        bn_momentum (float, optional):
            Momentum for batch norm layers.

        bn_epsilon (float, optional):
            Epsilon for batch norm layers.

        activation (:class:`torch.nn.Module`):
            Activation function to use on convolution layers.

        upsample_mode (str):
            Upsampling mode to use. See :class:`torch.nn.Upsample`

        checkpoint (bool):
            Whether or not to use gradient checkpointing. Checkpointing saves memory at the cost of
            added compute. See :func:`torch.utils.checkpoint.checkpoint` for more details.

    Shape:
        - Inputs: List of Tensors of shape :math:`(N, *C, *H, *W)` where :math:`*C, *H, *W` indicates
          variable channel/height/width at each level of downsapling.
        - Output: Same shape as input.

    .. _EfficientDet Scalable and Efficient Object Detection:
        https://arxiv.org/abs/1911.09070
    """


class BiFPN(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._bifpn = BiFPN2d(*args, **kwargs)
        import warnings

        warnings.warn("BiFPN is deprecated, please use BiFPN2d instead", category=DeprecationWarning)

    def forward(self, inputs: List[Tensor]) -> List[Tensor]:
        return self._bifpn(inputs)


class BiFPN3d(_BiFPN, metaclass=_BiFPNMeta):
    pass
