#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional, Tuple

import torch
from torch import Tensor
from torchvision.ops import box_iou


class ConfusionMatrixIoU:
    r"""Creates two boolean masks, one for true positivity of predicted boxes, and another for
    false negativity of target boxes. In the case of multiple predicted boxes correctly overlapping
    a target box, only the highest IoU box will be considered a true positive

    .. warning::
        This method is experimental

    Args:
        iou_threshold (float):
            Intersection over union threshold for which a box should be declared a positive

        true_positive_limit (bool):
            By default, if multiple predicted boxes correctly overlap a target box only one predicted box will be
            considered a true positive. If ``true_positive_limit=False``, consider all correctly overlapping boxes
            as true positives

    Returns:
        Tuple of ``(true_positive_mask, false_negative_mask)``

    Input Shape
        * ``pred_boxes`` - :math:`(N_p, 4)`
        * ``pred_classes`` - :math:`(N_p, 1)`
        * ``true_boxes`` - :math:`(N_t, 4)`
        * ``true_classes`` - :math:`(N_t, 1)`

    Output Shape
        * ``true_positive_mask`` - :math:`(N_p)`
        * ``false_negative_mask`` - :math:`(N_t)`
    """

    def __init__(self, iou_threshold: float = 0.5, true_positive_limit: bool = True):
        self.iou_threshold = float(abs(iou_threshold))
        if self.iou_threshold == 0:
            raise ValueError("Expected iou_threshold > 0")
        self.true_positive_limit = bool(true_positive_limit)

    def __call__(
        self,
        pred_boxes: Tensor,
        pred_classes: Tensor,
        true_boxes: Tensor,
        true_classes: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        self._validate_inputs(pred_boxes, pred_classes, true_boxes, true_classes)

        # init output buffers
        box_dim = -2
        pred_boxes.shape[box_dim]
        true_boxes.shape[box_dim]
        tp = torch.empty(pred_boxes.shape[-2], device=pred_boxes.device, dtype=torch.bool).fill_(False)
        fn = torch.empty(true_boxes.shape[-2], device=true_boxes.device, dtype=torch.bool).fill_(True)

        # get IoUs (IoU is set to 0 when below threshold or class mismatch)
        ious = self._get_ious(pred_boxes, pred_classes, true_boxes, true_classes)

        # get pred box index -> target box index mapping
        pred_box_index, target_box_index = ious.nonzero(as_tuple=True)
        pred_box_index.unsqueeze_(-1)
        target_box_index.unsqueeze_(-1)

        # create list of (pred_box_index, target_box_index, iou) sorted by descending IoU
        ious = ious[pred_box_index, target_box_index]
        iou_argsort_indices = ious.argsort(dim=0, descending=True).view(-1)
        pred_box_index = pred_box_index[iou_argsort_indices]
        target_box_index = target_box_index[iou_argsort_indices]

        # for multiple pred boxes -> 1 target box, keep only the highest IoU match
        if self.true_positive_limit:
            unique_target_boxes, unique_indices = target_box_index.unique(return_inverse=True)
            num_unique_target_boxes = unique_target_boxes.shape[0]
            final_mapping = torch.empty(num_unique_target_boxes, 2, device=unique_indices.device, dtype=torch.long)
            final_mapping[unique_indices, 0] = pred_box_index
            final_mapping[unique_indices, 1] = target_box_index
        else:
            final_mapping = torch.cat([pred_box_index, target_box_index], dim=-1)

        tp[final_mapping[..., 0]] = True
        fn[final_mapping[..., 1]] = False
        return tp, fn

    def _get_ious(
        self,
        pred_boxes: Tensor,
        pred_classes: Tensor,
        true_boxes: Tensor,
        true_classes: Tensor,
    ) -> Tensor:
        box_dim = -2
        num_pred_boxes = pred_boxes.shape[box_dim]
        num_true_boxes = true_boxes.shape[box_dim]

        # get IoU for each pred box w.r.t. true box
        iou = box_iou(pred_boxes, true_boxes)
        assert iou.shape[0] == pred_boxes.shape[box_dim]
        assert iou.shape[1] == true_boxes.shape[box_dim]

        # mask where where iou exceeds threshold
        iou_mask = iou >= self.iou_threshold

        # mask where pred class matches true class
        pred_class_mask = pred_classes.expand(-1, num_true_boxes)
        true_class_mask = true_classes.T.expand(num_pred_boxes, -1)
        class_mask = pred_class_mask == true_class_mask

        # logical and of IoU > threshold & pred_class = true_class
        iou_mask.logical_and_(class_mask)

        # set IoU for bad classes and IoU < threshold to zero
        iou[~iou_mask] = 0
        return iou

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
        num_boxes = [pred_num_boxes,] * 2 + [
            true_num_boxes,
        ] * 2

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


class BinaryLabelIoU(ConfusionMatrixIoU):
    r"""Given a set of predicted boxes (with scores and class labels) and a set of target boxes (with class labels),
    creates a mapping of predicted probabilities to target probabilities. This method is intented to take anchor box
    predictions and labels (particularly from CenterNet where overlap with a true box is not guaranteed) and produce
    an output that can be passed to metrics that expect a predicted score to true label association.

    .. warning::
        This method is experimental

    Args:
        iou_threshold (float):
            Intersection over union threshold for which a box should be declared a positive

        true_positive_limit (bool):
            By default, if multiple predicted boxes correctly overlap a target box only one predicted box will be
            considered a true positive. If ``true_positive_limit=False``, consider all correctly overlapping boxes
            as true positives

    Returns:
        Tuple of ``(predicted_score, true_binary_label)``

    Shape
        * ``pred_boxes`` - :math:`(N_p, 4)`
        * ``pred_scores`` - :math:`(N_p, 1)`
        * ``pred_classes`` - :math:`(N_p, 1)`
        * ``true_boxes`` - :math:`(N_t, 4)`
        * ``true_classes`` - :math:`(N_t, 1)`
    """

    def __init__(self, iou_threshold: float = 0.5, true_positive_limit: bool = True):
        super().__init__(iou_threshold, true_positive_limit)

    def __call__(
        self,
        pred_boxes: Tensor,
        pred_scores: Tensor,
        pred_classes: Tensor,
        true_boxes: Tensor,
        true_classes: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        tp, fn = super().__call__(pred_boxes, pred_classes, true_boxes, true_classes)

        num_pred_boxes = tp.numel()
        num_fn = fn.sum()

        pred = torch.empty(num_pred_boxes + num_fn, device=pred_boxes.device).type_as(pred_scores).fill_(0)
        target = torch.empty(num_pred_boxes + num_fn, device=pred_boxes.device).type_as(pred_scores).fill_(0)

        pred[:num_pred_boxes] = pred_scores.view(-1)
        target[:num_pred_boxes][tp] = 1
        target[num_pred_boxes:] = 1
        return pred, target


# TODO can this be merged with BinaryLabelIoU?
class CategoricalLabelIoU(ConfusionMatrixIoU):
    r"""Given a set of predicted boxes (with scores and class labels) and a set of target boxes (with class labels),
    creates a mapping of predicted probabilities to target probabilities. This method is intented to take anchor box
    predictions and labels (particularly from CenterNet where overlap with a true box is not guaranteed) and produce
    an output that can be passed to metrics that expect a predicted score to true label association.

    .. warning::
        This method is experimental

    Args:
        iou_threshold (float):
            Intersection over union threshold for which a box should be declared a positive

        true_positive_limit (bool):
            By default, if multiple predicted boxes correctly overlap a target box only one predicted box will be
            considered a true positive. If ``true_positive_limit=False``, consider all correctly overlapping boxes
            as true positives

    Returns:
        Tuple of ``(predicted_score, binary_target, box_type)``

    Shape
        * ``pred_boxes`` - :math:`(N_p, 4)`
        * ``pred_scores`` - :math:`(N_p, 1)`
        * ``pred_classes`` - :math:`(N_p, 1)`
        * ``true_boxes`` - :math:`(N_t, 4)`
        * ``true_classes`` - :math:`(N_t, 1)`
    """

    def __init__(self, iou_threshold: float = 0.5, true_positive_limit: bool = True):
        super().__init__(iou_threshold, true_positive_limit)

    def __call__(
        self,
        pred_boxes: Tensor,
        pred_scores: Tensor,
        pred_classes: Tensor,
        true_boxes: Tensor,
        true_classes: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        tp, fn = super().__call__(pred_boxes, pred_classes, true_boxes, true_classes)

        num_pred_boxes = tp.numel()
        num_fn = fn.sum()

        pred = pred_scores.new_zeros(num_pred_boxes + num_fn)
        target = pred_scores.new_empty(num_pred_boxes + num_fn).fill_(-1)
        binary_target = pred_scores.new_zeros(num_pred_boxes + num_fn, dtype=torch.uint8)

        pred[:num_pred_boxes] = pred_scores.view(-1)
        target[:num_pred_boxes][tp] = pred_classes[tp].view(-1)
        target[num_pred_boxes:] = true_classes[fn].view(-1)
        target[:num_pred_boxes][~tp] = pred_classes[~tp].view(-1)
        binary_target[:num_pred_boxes][tp] = 1.0
        binary_target[num_pred_boxes:] = 1.0
        return pred, binary_target, target
