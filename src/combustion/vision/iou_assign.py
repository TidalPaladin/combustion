#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional, Tuple

from torch import Tensor
from torchvision.ops import box_iou


class ConfusionMatrixIoU:
    r"""Creates two boolean masks, one for true positivity of predicted boxes, and another for
    false negativity of target boxes.

    .. warning::
        This method is experimental

    Args:
        iou_threshold (float):
            Intersection over union threshold for which a box should be declared a positive
    """

    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = float(abs(iou_threshold))
        if self.iou_threshold == 0:
            raise ValueError("Expected iou_threshold > 0")

    def __call__(
        self, pred_boxes: Tensor, pred_classes: Tensor, true_boxes: Tensor, true_classes: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        self._validate_inputs(pred_boxes, pred_classes, true_boxes, true_classes)

        # compute iou matrix of shape pred_boxes x true_boxes
        box_dim = -2
        num_pred_boxes = pred_boxes.shape[box_dim]
        num_true_boxes = true_boxes.shape[box_dim]
        iou = box_iou(pred_boxes, true_boxes)
        assert iou.shape[0] == pred_boxes.shape[box_dim]
        assert iou.shape[1] == true_boxes.shape[box_dim]

        iou_mask = iou >= self.iou_threshold
        pred_class_mask = pred_classes.expand(-1, num_true_boxes)
        true_class_mask = true_classes.T.expand(num_pred_boxes, -1)
        class_mask = pred_class_mask == true_class_mask

        _ = iou_mask & class_mask
        tp = _.any(dim=-1)
        fn = _.any(dim=-2).logical_not_()
        return tp, fn

    def _validate_inputs(
        self,
        pred_boxes: Tensor,
        pred_classes: Tensor,
        true_boxes: Tensor,
        true_classes: Tensor,
        pred_scores: Optional[Tensor] = None,
    ) -> None:
        names = ["pred_boxes", "pred_classes", "true_boxes", "true_classes"]
        tensors = [pred_boxes, pred_classes, true_boxes, true_classes]
        for name, tensor in zip(names, tensors):
            if tensor.ndim != 2:
                raise RuntimeError(f"expected {name}.ndim == 2 but found {tensor.ndim}")

        box_dim = -2
        pred_num_boxes = pred_boxes.shape[box_dim]
        true_num_boxes = true_boxes.shape[box_dim]

        last_dims = [4, 1, 4, 1]
        num_boxes = [pred_num_boxes,] * 2 + [true_num_boxes,] * 2

        for name, tensor, last_dim, num_box in zip(names, tensors, last_dims, num_boxes):
            if not tensor.shape[box_dim] == num_box:
                raise RuntimeError(f"bad num boxes in {name} -> expected {num_box}, found shape {tensor.shape}")
            if not tensor.shape[-1] == last_dim:
                raise RuntimeError(f"bad last dimension in {name} -> expected {last_dim}, found shape {tensor.shape}")

        if pred_scores is not None:
            if not pred_scores.shape[box_dim] == pred_num_boxes:
                raise RuntimeError(
                    f"bad num boxes in pred_scores -> expected {num_box}, found shape {pred_scores.shape}"
                )
