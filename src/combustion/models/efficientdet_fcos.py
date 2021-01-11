#!/usr/bin/env python
# -*- coding: utf-8 -*-


from typing import List, Optional, Tuple

import torch.nn as nn
from torch import Tensor

from combustion.nn import FCOSDecoder, MobileNetBlockConfig

from .efficientdet import EfficientDet2d


class EfficientDetFCOS(EfficientDet2d):
    def __init__(
        self,
        num_classes: int,
        block_configs: List[MobileNetBlockConfig],
        strides: List[int] = [8, 16, 32, 64, 128],
        fpn_levels: List[int] = [3, 5, 7, 8, 9],
        fpn_filters: int = 64,
        fpn_repeats: int = 3,
        width_coeff: float = 1.0,
        depth_coeff: float = 1.0,
        width_divisor: float = 8.0,
        min_width: Optional[int] = None,
        stem: Optional[nn.Module] = None,
        fpn_kwargs: dict = {},
        head_repeats: Optional[int] = None,
    ):
        super().__init__(
            block_configs,
            fpn_levels,
            fpn_filters,
            fpn_repeats,
            width_coeff,
            depth_coeff,
            width_divisor,
            min_width,
            stem,
            nn.Identity(),
            fpn_kwargs,
        )

        if head_repeats is None:
            if hasattr(self, "compound_coeff"):
                num_repeats = 3 + self.compound_coeff // 3
            else:
                num_repeats = 3
        else:
            num_repeats = head_repeats
        self.strides = strides
        self.fcos = FCOSDecoder(fpn_filters, num_classes, num_repeats, strides)

    def forward(self, inputs: Tensor) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
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
        output = self.fcos(output)
        return output

    @classmethod
    def from_predefined(cls, compound_coeff: int, num_classes: int, **kwargs) -> "EfficientDetFCOS":
        r"""Creates an EfficientDet model using one of the parameterizations defined in the
        `EfficientDet paper`_.

        Args:
            compound_coeff (int):
                Compound scaling parameter :math:`\phi`. For example, to construct EfficientDet-D0, set
                ``compound_coeff=0``.

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

        strides = [8, 16, 32, 64, 128]
        fpn_filters = int(64 * 1.35 ** compound_coeff)
        fpn_repeats = 3 + compound_coeff
        fpn_levels = [3, 5, 7, 8, 9]

        head_repeats = 4 + compound_coeff // 3

        final_kwargs = {
            "num_classes": num_classes,
            "block_configs": cls.DEFAULT_BLOCKS,
            "width_coeff": width_coeff,
            "depth_coeff": depth_coeff,
            "width_divisor": width_divisor,
            "strides": strides,
            "fpn_filters": fpn_filters,
            "fpn_repeats": fpn_repeats,
            "fpn_levels": fpn_levels,
            "head_repeats": head_repeats,
        }
        final_kwargs.update(kwargs)
        result = cls(**final_kwargs)
        result.compound_coeff = compound_coeff
        return result

    @staticmethod
    def create_boxes(*args, **kwargs):
        return FCOSDecoder.postprocess(*args, **kwargs)

    @staticmethod
    def reduce_heatmap(*args, **kwargs) -> Tensor:
        return FCOSDecoder.reduce_heatmaps(*args, **kwargs)
