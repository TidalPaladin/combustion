#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from combustion.nn import DynamicSamePad, MobileNetBlockConfig
from combustion.util import double, single, triple


class _EfficientNetMeta(type):
    def __new__(cls, name, bases, dct):
        x = super().__new__(cls, name, bases, dct)
        if "3d" in name:
            x.Conv = nn.Conv3d
            x.BatchNorm = nn.BatchNorm3d
            x._get_blocks = MobileNetBlockConfig.get_3d_blocks
            x.Tuple = triple
        elif "2d" in name:
            x.Conv = nn.Conv2d
            x.BatchNorm = nn.BatchNorm2d
            x._get_blocks = MobileNetBlockConfig.get_2d_blocks
            x.Tuple = double
        elif "1d" in name:
            x.Conv = nn.Conv1d
            x.BatchNorm = nn.BatchNorm1d
            x._get_blocks = MobileNetBlockConfig.get_1d_blocks
            x.Tuple = single
        else:
            raise RuntimeError(f"Metaclass: error processing name {cls.__name__}")
        return x


# inspired by https://github.com/lukemelas/EfficientNet-PyTorch/tree/master


class _EfficientNet(nn.Module):
    DEFAULT_BLOCKS = [
        MobileNetBlockConfig(
            kernel_size=3,
            num_repeats=1,
            input_filters=32,
            output_filters=16,
            expand_ratio=1,
            use_skipconn=True,
            stride=1,
            squeeze_excite_ratio=4,
            global_se=True,
            drop_connect_rate=0.2,
            bn_momentum=0.01,
            bn_epsilon=1e-3,
        ),
        MobileNetBlockConfig(
            kernel_size=3,
            num_repeats=2,
            input_filters=16,
            output_filters=24,
            expand_ratio=6,
            use_skipconn=True,
            stride=2,
            squeeze_excite_ratio=4,
            global_se=True,
            drop_connect_rate=0.2,
            bn_momentum=0.01,
            bn_epsilon=1e-3,
        ),
        MobileNetBlockConfig(
            kernel_size=5,
            num_repeats=2,
            input_filters=24,
            output_filters=40,
            expand_ratio=6,
            use_skipconn=True,
            stride=2,
            squeeze_excite_ratio=4,
            global_se=True,
            drop_connect_rate=0.2,
            bn_momentum=0.01,
            bn_epsilon=1e-3,
        ),
        MobileNetBlockConfig(
            kernel_size=3,
            num_repeats=3,
            input_filters=40,
            output_filters=80,
            expand_ratio=6,
            use_skipconn=True,
            stride=2,
            squeeze_excite_ratio=4,
            global_se=True,
            drop_connect_rate=0.2,
            bn_momentum=0.01,
            bn_epsilon=1e-3,
        ),
        MobileNetBlockConfig(
            kernel_size=5,
            num_repeats=3,
            input_filters=80,
            output_filters=112,
            expand_ratio=6,
            use_skipconn=True,
            stride=1,
            squeeze_excite_ratio=4,
            global_se=True,
            drop_connect_rate=0.2,
            bn_momentum=0.01,
            bn_epsilon=1e-3,
        ),
        MobileNetBlockConfig(
            kernel_size=5,
            num_repeats=4,
            input_filters=112,
            output_filters=192,
            expand_ratio=6,
            use_skipconn=True,
            stride=2,
            squeeze_excite_ratio=4,
            global_se=True,
            drop_connect_rate=0.2,
            bn_momentum=0.01,
            bn_epsilon=1e-3,
        ),
        MobileNetBlockConfig(
            kernel_size=3,
            num_repeats=1,
            input_filters=192,
            output_filters=320,
            expand_ratio=6,
            use_skipconn=True,
            stride=1,
            squeeze_excite_ratio=4,
            global_se=True,
            drop_connect_rate=0.2,
            bn_momentum=0.01,
            bn_epsilon=1e-3,
        ),
    ]

    def __init__(
        self,
        block_configs: List[MobileNetBlockConfig],
        width_coeff: float = 1.0,
        depth_coeff: float = 1.0,
        width_divisor: float = 8.0,
        min_width: Optional[int] = None,
        stem: Optional[nn.Module] = None,
        head: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.__block_configs = tuple(deepcopy(block_configs))
        self.__width_coeff = float(width_coeff)
        self.__depth_coeff = float(depth_coeff)

        output_filters = []
        for config in self.__block_configs:
            # update config according to scale coefficients
            config.input_filters = self.round_filters(config.input_filters, width_coeff, width_divisor, min_width)
            config.output_filters = self.round_filters(config.output_filters, width_coeff, width_divisor, min_width)
            config.num_repeats = self.round_repeats(depth_coeff, config.num_repeats)
            output_filters.append(config.output_filters)

        self.__input_filters = self.__block_configs[0].input_filters
        self.__output_filters = tuple(output_filters)

        # Conv stem (default stem used if none given)
        if stem is not None:
            self.stem = stem
        else:
            in_channels = 3
            first_block = next(iter(self.__block_configs))
            output_filters = first_block.input_filters
            bn_momentum = first_block.bn_momentum
            bn_epsilon = first_block.bn_epsilon

            self.stem = nn.Sequential(
                DynamicSamePad(self.Conv(in_channels, output_filters, kernel_size=3, stride=2, bias=False, padding=1)),
                self.BatchNorm(output_filters, momentum=bn_momentum, eps=bn_epsilon),
            )

        # MobileNetV3 convolution blocks
        blocks = []
        for config in self.__block_configs:
            conv_block = self.__class__._get_blocks(config)
            blocks.append(conv_block)
        self.blocks = nn.ModuleList(blocks)

        # Head
        self.head = head

    @torch.jit.unused
    @property
    def input_filters(self) -> int:
        r"""Number of input filters for the first level of backbone. When using a custom stem, use this
        property to determine the number of filters in the stem's output.
        """
        return self.__input_filters

    @torch.jit.unused
    @property
    def output_filters(self) -> Tuple[int, ...]:
        r"""Number of filters in each level of the BiFPN. When using a custom head, use this
        property to determine the number of filters in the head's input.
        """
        return self.__output_filters

    @torch.jit.unused
    @property
    def block_configs(self) -> Tuple[MobileNetBlockConfig, ...]:
        r"""Number of filters in each level of the BiFPN. When using a custom head, use this
        property to determine the number of filters in the head's input.
        """
        return self.__block_configs

    @torch.jit.unused
    @property
    def width_coeff(self) -> float:
        r"""Width coefficient for scaling"""
        return self.__width_coeff

    @torch.jit.unused
    @property
    def depth_coeff(self) -> float:
        r"""Depth coefficient for scaling"""
        return self.__depth_coeff

    def extract_features(self, inputs: Tensor, return_all: bool = False) -> List[Tensor]:
        r"""Runs the EfficientNet stem and body to extract features, returning a list of
        tensors representing features extracted from each block.

        Args:

            inputs (:class:`torch.Tensor`):
                Model inputs

            return_all (bool):
                By default, only features extracted from blocks with non-unit stride will
                be returned. If ``return_all=True``, return features extracted from every
                block group in the model.
        """
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
        r"""Runs the entire EfficientNet model, including stem, body, and head.
        If no head was supplied, the output of :func:`extract_features` will be returned.
        Otherwise, the output of the given head will be returned.

        .. note::
            The returned output will always be a list of tensors. If a custom head is given
            and it returns a single tensor, that tensor will be wrapped in a list before
            being returned.

        Args:
            inputs (:class:`torch.Tensor`):
                Model inputs

            return_all (bool):
                By default, only features extracted from blocks with non-unit stride will
                be returned. If ``return_all=True``, return features extracted from every
                block group in the model.
        """
        output = self.extract_features(inputs, use_all_features)
        if self.head is not None:
            output = self.head(output)
            if not isinstance(output, list):
                output = [
                    output,
                ]

        return output

    def round_filters(self, filters, width_coeff, width_divisor, min_width):
        if not width_coeff:
            return filters

        filters *= width_coeff
        min_width = min_width or width_divisor
        new_filters = max(min_width, int(filters + width_divisor / 2) // width_divisor * width_divisor)

        # prevent rounding by more than 10%
        if new_filters < 0.9 * filters:
            new_filters += width_divisor

        return int(new_filters)

    def round_repeats(self, depth_coeff, num_repeats):
        if not depth_coeff:
            return num_repeats
        return int(math.ceil(depth_coeff * num_repeats))

    @classmethod
    def from_predefined(cls, compound_coeff: int, block_overrides: Dict[str, Any] = {}, **kwargs) -> "_EfficientNet":
        r"""Creates an EfficientNet model using one of the parameterizations defined in the
        `EfficientNet paper`_.

        Args:
            compound_coeff (int):
                Compound scaling parameter :math:`\phi`. For example, to construct EfficientNet-B0, set
                ``compound_coeff=0``.

            block_overrides (dict):
                Overrides to be applied to each :class:`combustion.nn.MobileNetBlockConfig`.

            **kwargs:
                Additional parameters/overrides for model constructor.

        .. _EfficientNet paper:
            https://arxiv.org/abs/1905.11946
        """
        # from paper
        alpha = 1.2
        beta = 1.1
        width_divisor = 8.0

        depth_coeff = alpha ** compound_coeff
        width_coeff = beta ** compound_coeff

        # apply config overrides at each block
        block_configs = deepcopy(cls.DEFAULT_BLOCKS)
        for k, v in block_overrides.items():
            for config in block_configs:
                setattr(config, str(k), v)

        final_kwargs = {
            "block_configs": block_configs,
            "width_coeff": width_coeff,
            "depth_coeff": depth_coeff,
            "width_divisor": width_divisor,
        }
        final_kwargs.update(kwargs)
        model = cls(**final_kwargs)
        model.compound_coeff = compound_coeff
        return model

    def input_size(self, compound_coeff: Optional[int] = None) -> Tuple[int, ...]:
        r"""Returns the expected input size for a given compound coefficient.
        If a model was not created with :func:`from_predefined` a compound coefficient
        must be provided manually.

        Scaled resolution is computed as

        .. math::
            r = 512 + 128 * \phi

        Args:
            * compound_coeff (int):
                Compound coefficient :math:`\phi` to compute image size for.

        Returns:
            Tuple of ints giving the expected input spatial resolution
        """
        if compound_coeff is not None:
            x = int(compound_coeff)
        elif hasattr(self, "compound_coeff"):
            x = int(self.compound_coeff)
        else:
            raise ValueError("compound_coeff must be provided when model was not created via from_predefined()")
        return self.__class__.Tuple(512 + 128 * x)


class EfficientNet1d(_EfficientNet, metaclass=_EfficientNetMeta):
    pass


class EfficientNet2d(_EfficientNet, metaclass=_EfficientNetMeta):
    r"""Implementation of EfficientNet as described in the `EfficientNet paper`_.
    EfficientNet defines a family of models that are built using convolutional
    blocks that are parameterized such that width, depth, and spatial resolution
    can be easily scaled up or down. The authors of EfficientNet note that scaling
    each of these dimensions simultaneously is advantageous.

    Let the depth, width, and resolution of the model be denoted by :math:`d, w, r` respectively.
    EfficientNet defines a compound scaling factor :math:`\phi` such that

    .. math::
        d = \alpha^\phi \\
        w = \beta^\phi \\
        r = \gamma^\phi

    where

    .. math::
        \alpha * \beta^2 * \gamma^2 \approx 2

    The parameters :math:`\alpha,\beta,\gamma` are experimentally determined and describe
    how finite computational resources should be distributed amongst depth, width, and resolution
    scaling. The parameter :math:`\phi` is a user controllable compound scaling coefficient such that
    for a new :math:`\phi`, FLOPS will increase by approximately :math:`2^\phi`.


    The authors of EfficientNet selected the following scaling parameters:

    .. math::
        \alpha = 1.2 \\
        \beta = 1.1 \\
        \gamma = 1.15

    .. note::
        Currently, DropConnect ratios are not scaled based on depth of the given block.
        This is a deviation from the true EfficientNet implementation.

    Args:
        block_configs (list of :class:`combustion.nn.MobileNetBlockConfig`)
            Configs for each of the :class:`combustion.nn.MobileNetConvBlock2d` blocks
            used in the model.

        width_coeff (float):
            The width scaling coefficient. Increasing this increases the width of the model.

        depth_coeff (float):
            The depth scaling coefficient. Increasing this increases the depth of the model.

        width_divisor (float):
            Used in calculating number of filters under width scaling. Filters at each block
            will be a multiple of ``width_divisor``.

        min_width (int):
            The minimum width of the model at any block

        stem (:class:`torch.nn.Module`):
            An optional stem to use for the model. The default stem is a single
            3x3/2 conolution that expects 3 input channels.

        head (:class:`torch.nn.Module`):
            An optional head to use for the model. By default, no head will be used
            and ``forward`` will return a list of tensors containing extracted features.

    Shapes
        * Input: :math:`(N, C, H, W)`
        * Output: List of tensors of shape :math:`(N, C, H', W')`, where height and width vary
          depending on the amount of downsampling for that feature map.

    .. _EfficientNet paper:
        https://arxiv.org/abs/1905.11946
    """


class EfficientNet3d(_EfficientNet, metaclass=_EfficientNetMeta):
    pass
