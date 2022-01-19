#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Callable, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torchmetrics import MeanMetric as AverageMeter


class Entropy(AverageMeter):
    r"""Computes categorical or binary entropy over an input.
    Inputs are expected to be unnormalized logits (no Sigmoid/Softmax applied).

    Args:
        dim:
            Dimension to compute categorical entropy over. Should be ``None`` for binary entropy.

        compute_on_step:
            Forward only calls ``update()`` and return ``None`` if this is set to ``False``.
        dist_sync_on_step:
            Synchronize metric state across processes at each ``forward()``
            before returning the value at the step
        process_group:
            Specify the process group on which synchronization is called.
            default: ``None`` (which selects the entire world)
        dist_sync_fn:
            Callback that performs the allgather operation on the metric state. When ``None``, DDP
            will be used to perform the allgather
    """

    def __init__(
        self,
        dim: Optional[int] = None,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
        inplace: bool = True,
        from_logits: bool = True,
    ):
        super().__init__(
            compute_on_step,
            dist_sync_on_step,
            process_group,
            dist_sync_fn,
        )
        self.dim = int(dim) if dim is not None else None
        self.num_classes: Optional[int] = None
        self.inplace = inplace
        self.from_logits = from_logits

    def update(self, preds: Tensor) -> None:
        if self.dim is None:
            entropy = self.compute_binary_entropy(preds, self.inplace, self.from_logits)
        else:
            entropy = self.compute_categorical_entropy(
                preds, dim=self.dim, inplace=self.inplace, from_logits=self.from_logits
            )
        super().update(entropy)

    @staticmethod
    def compute_binary_entropy(x: Tensor, inplace: bool = True, from_logits: bool = True, eps: float = 1e6) -> Tensor:
        r"""Computes binary entropy pointwise over a tensor.
        Inputs are expected to be unnormalized logits (no Sigmoid applied).

        Args:
            x (:class:`torch.Tensor`):
                Tensor to compute binary entropy over
        """
        if from_logits:
            p = x.sigmoid()
            log_p = F.logsigmoid(x)
            log_p_inv = F.logsigmoid(x.neg())
        else:
            p = x
            log_p = x.log().clamp_min(-eps)
            log_p_inv = (1 - x).log().clamp_min(-eps)

        p_inv = 1 - p
        if inplace:
            return log_p.mul_(p).add_(log_p_inv.mul_(p_inv)).neg_()
        else:
            return log_p.mul(p).add(log_p_inv.mul(p_inv)).neg()

    @staticmethod
    def compute_categorical_entropy(
        x: Tensor, dim: int = -1, inplace: bool = True, from_logits: bool = True, eps: float = 1e6
    ) -> Tensor:
        r"""Computes categorical or binary entropy along a given tensor dimension. Inputs
        are expected to be unnormalized logits (no Softmax or Sigmoid applied).

        Args:
            x (:class:`torch.Tensor`):
                Tensor to compute entropy over

            dim (int):
                Dimension to compute entropy over. Only relevant for categorical entropy.
        """
        if from_logits:
            p = x.softmax(dim=dim)
            log_p = F.log_softmax(x, dim=dim)
        else:
            p = x
            log_p = x.log().clamp_min(-eps)

        C = x.shape[dim]
        assert C > 0

        with torch.no_grad():
            divisor = p.new_tensor(C).log_()

        if inplace:
            return log_p.mul_(p).sum(dim=dim).neg_().div_(divisor)
        else:
            return log_p.mul(p).sum(dim=dim).neg().div(divisor)
