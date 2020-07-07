#!/usr/bin/env python
# -*- coding: utf-8 -*-

from copy import deepcopy
from typing import List, Optional

import torch.nn as nn
from torch import Tensor

from combustion.nn import BiFPN1d, BiFPN2d, BiFPN3d, MobileNetBlockConfig

from .efficientnet import _EfficientNet


class _EfficientDetMeta(type):
    def __new__(cls, name, bases, dct):
        x = super().__new__(cls, name, bases, dct)
        if "3d" in name:
            x.Conv = nn.Conv3d
            x.BatchNorm = nn.BatchNorm3d
            x.BiFPN = BiFPN3d
            x._get_blocks = MobileNetBlockConfig.get_3d_blocks
        elif "2d" in name:
            x.Conv = nn.Conv2d
            x.BatchNorm = nn.BatchNorm2d
            x.BiFPN = BiFPN2d
            x._get_blocks = MobileNetBlockConfig.get_2d_blocks
        elif "1d" in name:
            x.Conv = nn.Conv1d
            x.BatchNorm = nn.BatchNorm1d
            x.BiFPN = BiFPN1d
            x._get_blocks = MobileNetBlockConfig.get_1d_blocks
        else:
            raise RuntimeError(f"Metaclass: error processing name {cls.__name__}")
        return x


class _EfficientDet(_EfficientNet):
    __constants__ = ["fpn_levels"]

    def __init__(
        self,
        block_configs: List[MobileNetBlockConfig],
        fpn_filters: int,
        fpn_levels: List[int],
        width_coeff: float = 1.0,
        depth_coeff: float = 1.0,
        width_divisor: float = 8.0,
        min_width: Optional[int] = None,
        stem: Optional[nn.Module] = None,
        head: Optional[nn.Module] = None,
    ):
        super(_EfficientDet, self).__init__(
            block_configs, width_coeff, depth_coeff, width_divisor, min_width, stem, head=None
        )
        self.fpn_levels = fpn_levels
        block_configs = deepcopy(block_configs)

        has_non_unit_stride = False
        for config in block_configs:
            # update config according to scale coefficients
            config.input_filters = self.round_filters(config.input_filters, width_coeff, width_divisor, min_width)
            config.output_filters = self.round_filters(config.output_filters, width_coeff, width_divisor, min_width)
            config.num_repeats = self.round_repeats(depth_coeff, config.num_repeats)
            has_non_unit_stride = has_non_unit_stride or config.stride > 1

        # convolutions mapping backbone feature maps to constant number of channels
        fpn_convs = []
        output_filters = self.round_filters(fpn_filters, width_coeff, width_divisor, min_width)
        for i, config in enumerate(block_configs):
            input_filters = config.output_filters
            conv = self.Conv(input_filters, output_filters, kernel_size=1)
            fpn_convs.append(conv)
        self.fpn_convs = nn.ModuleList(fpn_convs)

        bifpn_layers = []
        fpn_repeats = self.round_repeats(depth_coeff, len(fpn_levels) + 1)
        for i in range(fpn_repeats):
            bifpn = self.BiFPN(output_filters, levels=len(fpn_levels))
            bifpn_layers.append(bifpn)
        self.bifpn_layers = nn.ModuleList(bifpn_layers)

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
        captured_features: List[Tensor] = [backbone_features[i - 1] for c, i in enumerate(self.fpn_levels)]

        # map to constant channel number using trivial convs
        for i, conv in enumerate(self.fpn_convs):
            captured_features[i] = conv(captured_features[i])

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
            if not isinstance(output, list):
                output = [
                    output,
                ]

        return output


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

    .. note::
        Currently, DropConnect ratios are not scaled based on depth of the given block.
        This is a deviation from the true EfficientNet implementation.

    .. note::
        The number of BiFPN layers prior to depth scaling is chosen to be ``len(fpn_levels) - 1``
        such that information in the BiFPN will pass across all feature maps.

    Args:
        block_configs (list of :class:`combustion.nn.MobileNetBlockConfig`)
            Configs for each of the :class:`combustion.nn.MobileNetConvBlock2d` blocks
            used in the model.

        fpn_filters (int):
            Base number of filters to use for the BiFPN before width scaling.

        fpn_levels (list of ints):
            Indicies of EfficientNet feature levels to include in the BiFPN, starting at index 1.

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

    .. _EfficientDet paper:
        https://arxiv.org/abs/1911.09070
    """


class EfficientDet3d(_EfficientDet, metaclass=_EfficientDetMeta):
    pass
