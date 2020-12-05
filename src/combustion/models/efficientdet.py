#!/usr/bin/env python
# -*- coding: utf-8 -*-

from copy import deepcopy
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from combustion.nn import BiFPN1d, BiFPN2d, BiFPN3d, MatchShapes, MobileNetBlockConfig

from .efficientnet import _EfficientNet


class _EfficientDetMeta(type):
    def __new__(cls, name, bases, dct):
        x = super().__new__(cls, name, bases, dct)
        if "3d" in name:
            x.Conv = nn.Conv3d
            x.BatchNorm = nn.BatchNorm3d
            x.BiFPN = BiFPN3d
            x._get_blocks = MobileNetBlockConfig.get_3d_blocks
        elif "1d" in name:
            x.Conv = nn.Conv1d
            x.BatchNorm = nn.BatchNorm1d
            x.BiFPN = BiFPN1d
            x._get_blocks = MobileNetBlockConfig.get_1d_blocks
        else:
            x.Conv = nn.Conv2d
            x.BatchNorm = nn.BatchNorm2d
            x.BiFPN = BiFPN2d
            x._get_blocks = MobileNetBlockConfig.get_2d_blocks
        return x


class _EfficientDet(_EfficientNet):
    __constants__ = ["fpn_levels"]

    def __init__(
        self,
        block_configs: List[MobileNetBlockConfig],
        fpn_levels: List[int] = [3, 5, 7, 8, 9],
        fpn_filters: int = 64,
        fpn_repeats: int = 3,
        width_coeff: float = 1.0,
        depth_coeff: float = 1.0,
        width_divisor: float = 8.0,
        min_width: Optional[int] = None,
        stem: Optional[nn.Module] = None,
        head: Optional[nn.Module] = None,
        fpn_kwargs: dict = {},
    ):
        super(_EfficientDet, self).__init__(
            block_configs, width_coeff, depth_coeff, width_divisor, min_width, stem, head
        )
        self.fpn_levels = fpn_levels

        # convolutions mapping backbone feature maps to constant number of channels
        fpn_convs = []
        output_filters = self.round_filters(fpn_filters, 1.0, width_divisor, min_width)
        self.__fpn_filters = output_filters
        for i, config in enumerate(self.block_configs):
            if i + 1 in fpn_levels:
                input_filters = config.output_filters
                conv = self.Conv(input_filters, output_filters, kernel_size=1)
                fpn_convs.append(conv)

        for i in fpn_levels:
            if i == len(self.block_configs) + 1:
                input_filters = self.block_configs[-1].output_filters
                conv = self.Conv(input_filters, output_filters, kernel_size=3, stride=2, padding=1)
                fpn_convs.append(conv)
            elif i > len(self.block_configs) + 1:
                input_filters = output_filters
                conv = self.Conv(input_filters, output_filters, kernel_size=3, stride=2, padding=1)
                fpn_convs.append(nn.Sequential(nn.ReLU(), conv))

        self.fpn_convs = nn.ModuleList(fpn_convs)

        self.match = MatchShapes()

        # defaults for batch norm params
        _ = {"bn_momentum": 0.01, "bn_epsilon": 1e-3}
        _.update(fpn_kwargs)
        fpn_kwargs = _

        # build bifpn
        bifpn_layers = []
        for i in range(fpn_repeats):
            bifpn = self.BiFPN(output_filters, levels=len(fpn_levels), **fpn_kwargs)
            bifpn_layers.append(bifpn)
        self.bifpn_layers = nn.ModuleList(bifpn_layers)

    @torch.jit.unused
    @property
    def fpn_filters(self) -> int:
        r"""Number of filters in each level of the BiFPN. When using a custom head, use this
        property to determine the number of filters in the head's input.
        """
        return self.__fpn_filters

    def extract_features(self, inputs: Tensor) -> List[Tensor]:
        r"""Runs the EfficientDet stem and body to extract features, returning a list of
        tensors representing features extracted from each block.

        Args:

            inputs (:class:`torch.Tensor`):
                Model inputs

        """
        # efficientnet feature extractor
        backbone_features: List[Tensor] = []
        x = self.stem(inputs)
        prev_x = x
        for block in self.blocks:
            x = block(prev_x)
            backbone_features.append(x)
            prev_x = x

        # pull out feature maps to be used in BiFPN
        captured_features: List[Tensor] = []

        for i in self.fpn_levels:
            if i - 1 < len(backbone_features):
                captured_features.append(backbone_features[i - 1])

        # map to constant channel number using trivial convs
        for i, conv in enumerate(self.fpn_convs):
            if i < len(captured_features):
                captured_features[i] = conv(captured_features[i])
            else:
                prev_x = conv(prev_x)
                captured_features.append(prev_x)

        for bifpn in self.bifpn_layers:
            captured_features = bifpn(captured_features)

        return captured_features

    def forward(self, inputs: Tensor) -> List[Tensor]:
        r"""Runs the entire EfficientDet model, including stem, body, and head.
        If no head was supplied, the output of :func:`extract_features` will be returned.
        Otherwise, the output of the given head will be returned.

        .. note::
            The returned output will always be a list of tensors. If a custom head is given
            and it returns a single tensor, that tensor will be wrapped in a list before
            being returned.

        Args:
            inputs (:class:`torch.Tensor`):
                Model inputs
        """
        output = self.extract_features(inputs)
        if self.head is not None:
            output = self.head(output)
        return output

    @classmethod
    def from_predefined(cls, compound_coeff: int, block_overrides: Dict[str, Any] = {}, **kwargs) -> "_EfficientDet":
        r"""Creates an EfficientDet model using one of the parameterizations defined in the
        `EfficientDet paper`_.

        Args:
            compound_coeff (int):
                Compound scaling parameter :math:`\phi`. For example, to construct EfficientDet-D0, set
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

        fpn_filters = int(64 * 1.35 ** compound_coeff)
        fpn_repeats = 3 + compound_coeff
        fpn_levels = [3, 5, 7, 8, 9]

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
            "fpn_filters": fpn_filters,
            "fpn_repeats": fpn_repeats,
            "fpn_levels": fpn_levels,
        }
        final_kwargs.update(kwargs)
        result = cls(**final_kwargs)
        result.compound_coeff = compound_coeff
        return result


class EfficientDet1d(_EfficientDet, metaclass=_EfficientDetMeta):
    pass


class EfficientDet2d(_EfficientDet, metaclass=_EfficientDetMeta):
    r"""Implementation of EfficientDet as described in the `EfficientDet paper`_.
    EfficientDet is built on an EfficientNet backbone
    (see :class:`combustion.models.EfficientNet2d` for details). EfficientDet adds a
    bidirectional feature pyramid network (see :class:`combustion.nn.BiFPN2d`), which
    mixes information across the various feature maps produced by the EfficientNet backbone.

    .. image:: ./efficientdet.png
        :width: 800px
        :align: center
        :height: 300px
        :alt: Diagram of EfficientDet

    The authors of EfficientDet used the default EfficientNet scaling parameters for the backbone:

    .. math::
        \alpha = 1.2 \\
        \beta = 1.1 \\
        \gamma = 1.15


    The BiFPN was scaled as follows:

    .. math::
        W_\text{bifpn} = 64 \cdot \big(1.35^\phi\big) \\
        D_\text{bifpn} = 3 + \phi

    In the original EfficientDet implementation, the authors extract feature maps from levels
    3, 5, and 7 of the backbone. Two additional coarse levels are created by performing additional
    strided convolutions to the final level in the backbone, for a total of 5 levels in the BiFPN.

    .. note::
        Currently, DropConnect ratios are not scaled based on depth of the given block.
        This is a deviation from the true EfficientNet implementation.

    Args:
        block_configs (list of :class:`combustion.nn.MobileNetBlockConfig`)
            Configs for each of the :class:`combustion.nn.MobileNetConvBlock2d` blocks
            used in the model.

        fpn_levels (list of ints):
            Indicies of EfficientNet feature levels to include in the BiFPN, starting at index 1.
            Values in ``fpn_levels`` greater than the total number of blocks in the backbone denote
            levels that should be created by applying additional strided convolutions to the final
            level in the backbone.

        fpn_filters (int):
            Number of filters to use for the BiFPN. The filter count given here should be the desired
            number of filters after width scaling.

        fpn_repeats (int):
            Number of repeats to use for the BiFPN. The repeat count given here should be the desired
            number of repeats after depth scaling.

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

        fpn_kwargs (dict):
            Keyword args to be passed to all :class:`combustion.nn.BiFPN2d` layers.

    Shapes
        * Input: :math:`(N, C, H, W)`
        * Output: List of tensors of shape :math:`(N, C, H', W')`, where height and width vary
          depending on the amount of downsampling for that feature map.

    .. _EfficientDet paper:
        https://arxiv.org/abs/1911.09070
    """


class EfficientDet3d(_EfficientDet, metaclass=_EfficientDetMeta):
    pass
