#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Optional

import torch
from pytorch_lightning.metrics import Metric
from pytorch_lightning.metrics.classification.precision_recall import _input_format
from pytorch_lightning.metrics.functional.reduction import class_reduce


class Fbeta(Metric):
    """
    Computes the Fbeta metric. This is equivalent to Pytorch Lightning's
    Fbeta metric, but includes the option to compute a weighted average

    .. note::
        This implementation will be removed once Pytorch Lightning's Fbeta
        metric supports weighted averaging

    """

    def __init__(
        self,
        num_classes: int = 1,
        beta: float = 1.0,
        threshold: float = 0.5,
        average: str = "micro",
        multilabel: bool = False,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
        )

        self.num_classes = num_classes
        self.beta = beta
        self.threshold = threshold
        self.average = average
        self.multilabel = multilabel

        assert self.average in (
            "micro",
            "macro",
            "weighted",
        ), "average passed to the function must be either `micro`, `macro`, or `weighted`"

        self.add_state("true_positives", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("predicted_positives", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("actual_positives", default=torch.zeros(num_classes), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        preds, target = _input_format(self.num_classes, preds, target, self.threshold, self.multilabel)

        self.true_positives += torch.sum(preds * target, dim=1)
        self.predicted_positives += torch.sum(preds, dim=1)
        self.actual_positives += torch.sum(target, dim=1)

    def compute(self):
        """
        Computes accuracy over state.
        """
        if self.average == "micro":
            precision = self.true_positives.sum().float() / (self.predicted_positives.sum())
            recall = self.true_positives.sum().float() / (self.actual_positives.sum())

        elif self.average in ["macro", "weighted"]:
            precision = self.true_positives.float() / (self.predicted_positives)
            recall = self.true_positives.float() / (self.actual_positives)

        num = (1 + self.beta ** 2) * precision * recall
        denom = self.beta ** 2 * precision + recall

        if self.average == "weighted":
            weights = self.actual_positives
            class_reduction = "weighted"
        else:
            weights = None
            class_reduction = "macro"

        return class_reduce(num=num, denom=denom, weights=weights, class_reduction=class_reduction)
