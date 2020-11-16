#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch import Tensor
import torch

from combustion.vision.centernet import CenterNetMixin
from typing import Dict, Tuple, List, Union, Optional

from pytorch_lightning.metrics.functional.classification import auroc as compute_auroc
from pytorch_lightning.metrics.functional.classification import average_precision as compute_ap


class MetricMixin:
    @staticmethod
    def get_global_pred_target_pairs_on_batch(
        type_heatmap: Tensor, tar_bbox: Tensor, tar_type: Optional[Tensor], pad_value: float = -1, **kwargs
    ) -> Tensor:
        # compute pred target pairs for types
        target = CenterNetMixin.combine_box_target(tar_bbox, tar_type)
        type_pairs = CenterNetMixin.get_global_pred_target_pairs(type_heatmap, target, pad_value=pad_value, **kwargs)
        return type_pairs

    @staticmethod
    def get_local_pred_target_pairs_on_batch(
        type_heatmap: Tensor, tar_bbox: Tensor, tar_type: Tensor, pad_value: float = -1, **kwargs
    ) -> Union[Tensor, List[Tensor]]:
        target = CenterNetMixin.combine_box_target(tar_bbox, tar_type)
        type_pairs = CenterNetMixin.get_pred_target_pairs(type_heatmap, target, upsample, pad_value=pad_value, **kwargs)
        return type_pairs

    # TODO replace with combution implementation
    @staticmethod
    def filter_classes(
        keep_class: int,
        type_heatmap: Tensor,
        tar_bbox: Tensor,
        tar_type: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        r"""Given a set of predicted heatmaps and target bounding boxes, filter the predictions and targets
        to include only a given class
        """
        # copy targets
        tar_bbox = tar_bbox.clone()
        tar_type = tar_type.clone()

        # mask of classes to keep
        keep_targets = (tar_type == keep_class).all(dim=-1)

        # fill non-keep locations
        tar_bbox[~keep_targets] = -1
        tar_type[~keep_targets] = -1

        # filter heatmap to class channels
        type_heatmap = torch.cat(
            [type_heatmap[..., keep_class : keep_class + 1, :, :], type_heatmap[..., -4:, :, :]], dim=-3
        )

        return type_heatmap, tar_bbox, tar_type
