#!/usr/bin/env python
# -*- coding: utf-8 -*-


import torch
from torch import Tensor
from torchmetrics import Metric

from .entropy import Entropy


@torch.no_grad()
def assign_to_bin(confidence: Tensor, num_bins: int) -> Tensor:
    thresholds = torch.linspace(0, 1, num_bins + 1, device=confidence.device, dtype=confidence.dtype)[1:]
    bins = confidence.unsqueeze(-1).lt(thresholds).long().argmax(dim=-1)
    return bins


class ECE(Metric):
    r"""Computes the Expected Calibration Error (ECE) between a set of predictions and ground truth.
    ECE is a measure of how well a model's predicted PMF matches the observed accuracy.
    """
    correct: Tensor
    confidence: Tensor
    total: Tensor

    def __init__(
        self,
        num_bins: int,
        num_classes: int = 1,
        threshold: float = 0.5,
        classwise: bool = False,
        from_logits: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        assert num_bins > 1
        assert 0 < threshold < 1
        self.num_bins = num_bins
        self.from_logits = from_logits
        self.threshold = threshold
        self.classwise = classwise
        self.num_classes = num_classes

        shape = (self.num_bins,) if not classwise else (self.num_bins, num_classes)
        self.add_state("confidence", default=torch.zeros(*shape, dtype=torch.float), dist_reduce_fx="sum")
        self.add_state("correct", default=torch.zeros(*shape, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("total", default=torch.zeros(*shape, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, pred: Tensor, true: Tensor) -> None:
        if pred.shape == true.shape:
            self.update_binary(pred, true)
        else:
            self.update_categorical(pred, true)

    def compute(self) -> Tensor:
        accuracy = self.correct / self.total.clamp_min(1)
        confidence = self.confidence / self.total.clamp_min(1)
        weights = self.total / self.total.sum(dim=0)

        ece = (accuracy - confidence).abs().mul(weights).sum(dim=0).mean(dim=0)
        return ece

    def _get_confidence(self, pred: Tensor, categorical: bool) -> Tensor:
        if not categorical:
            return pred
        return pred.amax(dim=-1)

    def update_binary(self, pred: Tensor, true: Tensor) -> None:
        if self.from_logits:
            pred = pred.sigmoid()

        pred_cls = pred >= self.threshold
        correct = (pred_cls == true).type_as(self.correct)
        confidence = self._get_confidence(pred, categorical=False)
        bins = assign_to_bin(confidence, self.num_bins).long()

        total = torch.zeros_like(self.total)
        idx, tot = bins.unique(return_counts=True)
        total[idx] = tot.type_as(total)

        self.correct = self.correct.scatter_add(0, bins.view(-1), correct.view(-1))
        self.confidence = self.confidence.scatter_add(0, bins.view(-1), confidence.view(-1))
        self.total = self.total + total

    def update_categorical(self, pred: Tensor, true: Tensor) -> None:
        if self.from_logits:
            pred = pred.softmax(dim=-1)

        pred_cls = pred.argmax(dim=-1)
        correct = (pred_cls == true).type_as(self.correct)
        confidence = self._get_confidence(pred, categorical=True)
        bins = assign_to_bin(confidence, self.num_bins).long()
        bins = bins * self.num_classes + true if self.classwise else bins

        total = torch.zeros_like(self.total)
        idx, tot = bins.unique(return_counts=True)
        total.view(-1)[idx] = tot.type_as(total)

        self.correct = self.correct.view(-1).scatter_add(0, bins.view(-1), correct.view(-1)).view_as(self.correct)
        self.confidence = (
            self.confidence.view(-1).scatter_add(0, bins.view(-1), confidence.view(-1)).view_as(self.confidence)
        )
        self.total = self.total + total

    @property
    def is_differentiable(self) -> bool:
        return False


class UCE(ECE):
    correct: Tensor
    confidence: Tensor
    total: Tensor

    def __init__(
        self,
        num_bins: int,
        num_classes: int = 1,
        threshold: float = 0.5,
        classwise: bool = False,
        from_logits: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(num_bins, num_classes, threshold, classwise, from_logits, *args, **kwargs)

    def compute(self) -> Tensor:
        err = 1 - self.correct / self.total.clamp_min(1)
        entropy = 1 - self.confidence / self.total.clamp_min(1)
        weights = self.total / self.total.sum(dim=0)

        uce = (err - entropy).abs().mul(weights).sum(dim=0).mean(dim=0)
        return uce

    def _get_confidence(self, pred: Tensor, categorical: bool) -> Tensor:
        if not categorical:
            conf = 1 - Entropy.compute_binary_entropy(pred, inplace=False)
        else:
            conf = 1 - Entropy.compute_categorical_entropy(pred, dim=-1, inplace=False)
        assert (conf >= 0).all()
        assert (conf <= 1).all()
        return conf
