#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Callable, Optional

import torch
from pytorch_lightning.metrics import Metric
from pytorch_lightning.metrics.functional import auroc
from torch import Tensor

from .bbox_metric import BoxClassificationMetric


class AUROC(Metric):
    r"""Computes Area Under the Receiver Operating Characteristic Curve (AUROC)
    from prediction scores.

    Args:
        compute_on_step:
            Forward only calls ``update()`` and returns None if this is set to False. default: True
        dist_sync_on_step:
            Synchronize metric state across processes at each ``forward()``
            before returning the value at the step. default: False
        process_group:
            Specify the process group on which synchronization is called. default: None (which selects the entire world)
    """

    def __init__(
        self,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
        pos_label: int = 1,
    ):
        super().__init__(compute_on_step, dist_sync_on_step, process_group, dist_sync_fn)

        self.pos_label = int(pos_label)
        self.add_state("pred", default=torch.empty(0), dist_reduce_fx="cat")
        self.add_state("true", default=torch.empty(0), dist_reduce_fx="cat")

    def update(self, preds: Tensor, target: Tensor):
        assert preds.shape == target.shape
        self.pred = torch.cat([self.pred, preds])
        self.true = torch.cat([self.true, target])

    def compute(self):
        return auroc(self.pred, self.true, pos_label=self.pos_label)


class BoxAUROC(BoxClassificationMetric):
    r"""Computes Area Under the Receiver Operating Characteristic Curve (AUROC)
    using anchor boxes.

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

    def compute(self):
        if self.binary_target.sum() == self.binary_target.numel():
            return torch.tensor(float("nan"), device=self.pred_score.device)

        return auroc(self.pred_score, self.binary_target)
