#!/usr/bin/env python
# -*- coding: utf-8 -*-


from abc import ABC
from typing import Any, Callable, Optional

import torch
from torch import Tensor
from torchmetrics import Metric

from combustion.util import check_dimension, check_is_tensor, check_ndim_match
from combustion.vision import CategoricalLabelIoU, split_bbox_scores_class, split_box_target


class BoxClassificationMetric(Metric, ABC):
    r"""Base class for classification metrics that involve anchor boxes.
    This implementation provides an update method that collects binary prediction
    scores, object type, and binary targets for each predicted/target anchor box.

    Args:
        pos_label (float, optional):
            Label type to compute the metric for. Boxes that are not of ``pos_label`` type
            will be discarded. By default all boxes are retained.

        iou_threshold (float):
            Intersection over union threshold for a prediction to be considered a true
            positive.

        true_positive_limit (bool):
            If ``False``, consider all boxes with IoU above threshold with a target box as
            true positives.

        pred_box_limit (int, optional):
            If given, only include the top ``pred_box_limit`` predicted boxes (by score)
            in the calculation.

        compute_on_step: See :class:`pytorch_lightning.metrics.Metric`
        dist_sync_on_step: See :class:`pytorch_lightning.metrics.Metric`
        process_group: See :class:`pytorch_lightning.metrics.Metric`
        dist_sync_fn: See :class:`pytorch_lightning.metrics.Metric`

    Shapes:
        * ``pred`` - :math:`(N, 6)` in form :math:`(x_1, y_1, x_2, y_2, \text{score}, \text{type})`
        * ``target`` - :math:`(N, 5)` in form :math:`(x_1, y_1, x_2, y_2, \text{type})`
    """

    def __init__(
        self,
        pos_label: Optional[float] = None,
        iou_threshold: float = 0.5,
        true_positive_limit: bool = True,
        pred_box_limit: Optional[int] = None,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
        )

        self.pos_label = float(pos_label) if pos_label is not None else None
        self.iou_threshold = float(iou_threshold)
        self.true_positive_limit = bool(true_positive_limit)
        self.pred_box_limit = pred_box_limit

        self.add_state("pred_score", default=torch.empty(0), dist_reduce_fx="cat")
        self.add_state("target_class", default=torch.empty(0), dist_reduce_fx="cat")
        self.add_state("binary_target", default=torch.empty(0), dist_reduce_fx="cat")

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        # preds, target = _input_format(self.num_classes, preds, target, self.threshold, self.multilabel)
        check_is_tensor(pred, "pred")
        check_is_tensor(target, "target")
        check_ndim_match(pred, target, "pred", "target")
        check_dimension(pred, -1, 6, "pred")
        check_dimension(target, -1, 5, "pred")

        # restrict the number of predicted boxes to the top K highest confidence boxes
        if self.pred_box_limit is not None and pred.shape[-2] > self.pred_box_limit:
            indices = pred[..., -2].argsort()
            pred = pred[indices, ...]
            assert pred.shape[-2] <= self.pred_box_limit

        # restrict pred and target to class of interest
        if self.pos_label is not None:
            pred_keep = pred[..., -1] == self.pos_label
            pred = pred[pred_keep]
            target_keep = target[..., -1] == self.pos_label
            target = target[target_keep]

        pred_score, target_class, binary_target = self.get_pred_target_pairs(pred, target)
        self.pred_score = torch.cat([self.pred_score, pred_score])
        self.target_class = torch.cat([self.target_class, target_class])
        self.binary_target = torch.cat([self.binary_target, binary_target])

    def get_pred_target_pairs(
        self,
        pred: Tensor,
        target: Tensor,
    ) -> Tensor:
        r"""Given a predicted CenterNet heatmap and target bounding box label, use box IoU to
        create a paring of predicted and target boxes such that each predicted box has
        an associated gold standard label.

        .. warning::
            This method should work with batched input, but such inputs are not thoroughly tested

        Args:
            pred (:class:`torch.Tensor`):
                Predicted heatmap.

            target (:class:`torch.Tensor`):
                Target bounding boxes in format ``x1, y1, x2, y2, class``.

            iou_threshold (float):
                Intersection over union threshold for which a prediction can be considered a
                true positive.

            true_positive_limit (bool):
                By default, only one predicted box overlapping a target box will be counted
                as a true positive. If ``False``, allow multiple true positive boxes per
                target box.

        Returns:
            A 3-tuple of tensors as follows:
                1. Predicted binary score for each box
                2. Classification target value for each box
                3. Boolean indicating if the prediction was a correct

        Shape:
            * ``pred`` - :math:`(N_{pred}, 6)`
            * ``target`` - :math:`(N_{true}, 5)`
            * Output - :math:`(N_o)`, :math:`(N_o)`, :math:`(N_o)`
        """
        # get a paring of predicted probability to target labels
        # if we didn't detect a target box at any threshold, assume P_pred = 0.0
        xform = CategoricalLabelIoU(self.iou_threshold, self.true_positive_limit)
        pred_boxes, pred_scores, pred_cls = split_bbox_scores_class(pred)
        target_bbox, target_cls = split_box_target(target)
        pred_out, binary_target, target_out = xform(pred_boxes, pred_scores, pred_cls, target_bbox, target_cls)

        assert pred_out.ndim == 1
        assert target_out.ndim == 1
        assert pred_out.shape == target_out.shape
        return pred_out, target_out.long(), binary_target
