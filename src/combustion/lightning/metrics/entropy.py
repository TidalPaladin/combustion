#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Callable, Optional

import torch.nn.functional as F
from torch import Tensor
from torchmetrics import AverageMeter


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
    ):
        super().__init__(
            compute_on_step,
            dist_sync_on_step,
            process_group,
            dist_sync_fn,
        )
        self.dim = int(dim) if dim is not None else None
        self.num_classes: Optional[int] = None

    def update(self, preds: Tensor) -> None:
        if self.dim is None:
            entropy = self.compute_binary_entropy(preds)
        else:
            entropy = self.compute_categorical_entropy(preds, dim=self.dim)
        super().update(entropy)

    @staticmethod
    def compute_binary_entropy(x: Tensor) -> Tensor:
        r"""Computes binary entropy pointwise over a tensor.
        Inputs are expected to be unnormalized logits (no Sigmoid applied).

        Args:
            x (:class:`torch.Tensor`):
                Tensor to compute binary entropy over
        """
        p = x.sigmoid()
        log_p = F.logsigmoid(x)
        p_inv = 1 - p
        log_p_inv = F.logsigmoid(x.neg())
        return log_p.mul_(p).add_(log_p_inv.mul_(p_inv)).neg_()

    @staticmethod
    def compute_categorical_entropy(x: Tensor, dim: int = -1) -> Tensor:
        r"""Computes categorical or binary entropy along a given tensor dimension. Inputs
        are expected to be unnormalized logits (no Softmax or Sigmoid applied).

        Args:
            x (:class:`torch.Tensor`):
                Tensor to compute entropy over

            dim (int):
                Dimension to compute entropy over. Only relevant for categorical entropy.
        """
        p = x.softmax(dim=dim)
        log_p = F.log_softmax(x, dim=dim)
        return log_p.mul_(p).sum(dim=dim).neg_()
