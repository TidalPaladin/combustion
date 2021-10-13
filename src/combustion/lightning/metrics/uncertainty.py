#!/usr/bin/env python
# -*- coding: utf-8 -*-


from typing import Optional, Tuple

import matplotlib.pyplot as plt
import torch
from torch import Tensor
from torchmetrics import Metric

from .entropy import Entropy


@torch.no_grad()
def assign_to_bin(confidence: Tensor, num_bins: int) -> Tensor:
    thresholds = torch.linspace(0, 1, num_bins + 1, device=confidence.device, dtype=confidence.dtype)[1:]
    bins = confidence.unsqueeze(-1).le(thresholds).long().argmax(dim=-1)
    return bins


class ECE(Metric):
    r"""Computes the Expected Calibration Error (ECE) between a set of predictions and ground truth.
    ECE is a measure of how well a model's predicted PMF matches the observed accuracy. ECE is defined as

    .. math::
        ECE = \sum_{m=1}^M \frac{\left\vert B_m \right\vert}{n} \left\vert acc(B_m) - conf(B_m)\right\vert

    ECE can also be computed classwise as

    .. math::
        cECE = \sum_{c=1}^C ECE(c)

    Args:
        num_bins:
            Number of bins :math:`M` in the calculation

        num_classes:
            Number of classes (for categorical classification)

        threshold:
            Threshold for binary classification

        classwise:
            If ``True``, compute classwise ECE (cECE)

        from_logits:
            If ``True``, expect the inputs to be unnormalized logits
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
            conf = pred.sigmoid() if self.from_logits else pred
        else:
            conf = pred.amax(dim=-1) if self.from_logits else pred.softmax(dim=-1)
        conf = conf.clamp(min=0, max=1)
        return conf

    def update_binary(self, pred: Tensor, true: Tensor) -> None:
        if self.from_logits:
            pred = pred.sigmoid()
        else:
            assert (pred >= 0).all() and (pred <= 1).all()

        pred_cls = pred >= self.threshold
        correct = (pred_cls == true).type_as(self.correct)
        confidence = self._get_confidence(pred, categorical=False)
        bins = assign_to_bin(confidence, self.num_bins).long()

        total = torch.zeros_like(self.total)
        idx, tot = bins.unique(return_counts=True)
        total[idx] = tot.type_as(total)

        self.correct = self._scatter_add(self.correct, bins, correct)
        self.confidence = self._scatter_add(self.confidence, bins, confidence)
        self.total = self.total + total

    def update_categorical(self, pred: Tensor, true: Tensor) -> None:
        if self.from_logits:
            pred = pred.softmax(dim=-1)
        else:
            s = pred.sum(dim=-1)
            assert torch.allclose(s, torch.ones_like(s))
            del s

        pred_cls = pred.argmax(dim=-1)
        correct = (pred_cls == true).type_as(self.correct)
        confidence = self._get_confidence(pred, categorical=True)
        bins = assign_to_bin(confidence, self.num_bins).long()
        bins = bins * self.num_classes + true if self.classwise else bins

        total = torch.zeros_like(self.total)
        idx, tot = bins.unique(return_counts=True)
        total.view(-1)[idx] = tot.type_as(total)

        self.correct = self._scatter_add(self.correct, bins, correct)
        self.confidence = self._scatter_add(self.confidence, bins, confidence)
        self.total = self.total + total

    @property
    def is_differentiable(self) -> bool:
        return False

    @staticmethod
    def _scatter_add(src: Tensor, idx: Tensor, vals: Tensor) -> Tensor:
        return (
            src.view(1, -1)
            .scatter_add(0, idx.view(-1), vals.view(-1))
            .view_as(src)
        )


class UCE(ECE):
    r"""Computes the Expected Uncertainty Calibration Error (UCE) between a set of predictions and ground truth.
    UCE is a measure of how well a model's predicted PMF matches the observed error rate. UCE was defined in the paper
    `Calibration of Model Uncertainty for Dropout Variational Inference`_.

    UCE is defined as

    .. math::
        UCE = \sum_{m=1}^M \frac{\left\vert B_m \right\vert}{n} \left\vert err(B_m) - uncert(B_m)\right\vert

    where :math:`uncert` is the normalized entropy per bin.

    UCE can also be computed classwise as

    .. math::
        cUCE = \sum_{c=1}^C UCE(c)

    Args:
        num_bins:
            Number of bins :math:`M` in the calculation

        num_classes:
            Number of classes (for categorical classification)

        threshold:
            Threshold for binary classification

        classwise:
            If ``True``, compute classwise UCE (cUCE)

        from_logits:
            If ``True``, expect the inputs to be unnormalized logits

    .. _Calibration of Model Uncertainty for Dropout Variational Inference:
        https://arxiv.org/abs/2006.11584
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
        super().__init__(num_bins, num_classes, threshold, classwise, from_logits, *args, **kwargs)

    def compute(self) -> Tensor:
        err = 1 - self.correct / self.total.clamp_min(1)
        entropy = 1 - self.confidence / self.total.clamp_min(1)
        weights = self.total / self.total.sum(dim=0)

        uce = (err - entropy).abs().mul(weights).sum(dim=0).mean(dim=0)
        return uce

    def _get_confidence(self, pred: Tensor, categorical: bool) -> Tensor:
        if not categorical:
            conf = 1 - Entropy.compute_binary_entropy(pred, inplace=False, from_logits=False)
        else:
            conf = 1 - Entropy.compute_categorical_entropy(pred, dim=-1, inplace=False, from_logits=False)
        conf = conf.clamp(min=0, max=1)
        return conf


class ErrorAtUncertainty(UCE):
    r"""Performs the UCE calculation, but returns the error and uncertainty for each bin without reduction.
    The output of this metric is sufficient for generating a plot of error rate vs uncertainty.

    .. image:: ./erroratuncert.png
        :width: 600px
        :align: center
        :height: 400px
        :alt: Sample plot generated by this metric.

    .. note:
        This metric is not well defined when ``classwise=True``, since classwise reduction would normally
        happen after reducing across all bins. In the classwise case, this metric will compute a per-bin
        mean across all classes.

    Args:
        num_bins:
            Number of bins :math:`M` in the calculation

        num_classes:
            Number of classes (for categorical classification)

        threshold:
            Threshold for binary classification

        classwise:
            If ``True``, compute classwise UCE (cUCE)

        from_logits:
            If ``True``, expect the inputs to be unnormalized logits
    """

    def compute(self) -> Tuple[Tensor, Tensor, Tensor]:
        err = 1 - self.correct / self.total.clamp_min(1)
        entropy = 1 - self.confidence / self.total.clamp_min(1)
        has_items = self.total.bool()

        err[~has_items] = 0
        entropy[~has_items] = 0

        if self.classwise:
            err = err.mean(dim=-1)
            entropy = entropy.mean(dim=-1)
            has_items = has_items.any(dim=-1)

        return entropy, err, has_items

    @staticmethod
    def plot(entropy: Tensor, err: Tensor, ax: Optional[plt.Axes] = None) -> Optional[plt.Figure]:
        r"""Generates a plot of error rate vs uncertainty.

        Args:
            entropy:
                Entropy as computed by the metric

            err:
                Error rate as computed by the metric

            ax:
                Pyplot Axes on which to generate the plot. By default, generate a suitable figure/axes using
                :func:`ErrorAtUncertainty.create_fig`

        Returns:
            :class:`matplotlib.pyplot.Figure` for the generated figure if ``ax`` was not provided, otherwise
            ``None``.
        """
        if ax is None:
            fig, ax = ErrorAtUncertainty.create_fig()
        else:
            fig = None
        ax.plot(entropy.cpu(), err.cpu(), marker="o")
        return fig

    @staticmethod
    def create_fig(**kwargs) -> Tuple[plt.Figure, plt.Axes]:
        r"""Sets up a Pyplot figure and axes for generating a plot of uncertainty vs error rate.

        Keyword Args:
            Forwarded to :func:`matplotlib.pyplot.figure`

        Returns:
            Tuple of :class:`plt.Figure`, :class:`plt.Axes`.
        """
        fig = plt.figure(**kwargs)
        ax: plt.Axes = fig.add_subplot(111)  # type: ignore
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Uncertainty")
        ax.set_ylabel("Error Rate")
        ax.plot([0, 1], [0, 1], "--", color="black", transform=ax.transAxes)
        ax.grid()
        return fig, ax
