#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torchmetrics import MeanMetric as AverageMeter


class Entropy(AverageMeter):
    r"""Computes normalized categorical or binary entropy over an input.

    .. note:
        Although this metric supports inputs as logits or normalized probabilities, the computation
        will likely be more accurate when using logits as input.

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

        from_logits:
            If ``True``, assume inputs will be unnormalized logits. Otherwise, assume inputs
            are normalized probabilities.

        eps:
            Numerical stabilizer for non-logit inputs
    """

    def __init__(
        self,
        dim: Optional[int] = None,
        compute_on_step: bool = True,
        from_logits: bool = True,
        eps: float = 1e6,
        **kwargs,
    ):
        super().__init__("ignore", compute_on_step, **kwargs)
        self.dim = int(dim) if dim is not None else None
        self.num_classes: Optional[int] = None
        self.from_logits = from_logits
        self.eps = eps

    def log(self, x: Tensor) -> Tensor:
        return x.log2() if self.bits else x.log()

    def log_(self, x: Tensor) -> Tensor:
        return x.log2_() if self.bits else x.log_()

    def update(self, preds: Tensor) -> None:
        if self.dim is None:
            entropy = self.compute_binary_entropy(preds, from_logits=self.from_logits, eps=self.eps)
        else:
            entropy = self.compute_categorical_entropy(preds, dim=self.dim, from_logits=self.from_logits, eps=self.eps)
        super().update(entropy)

    @staticmethod
    def compute_binary_entropy(x: Tensor, from_logits: bool = True, eps: float = 1e6) -> Tensor:
        r"""Computes normalized binary entropy pointwise over a tensor. The output is
        dimensionless (i.e. not in nats or bits) because the output is normalized.

        Args:
            x:
                Tensor to compute binary entropy over

            dim:
                Dimension to compute entropy over. Only relevant for categorical entropy.

            from_logits:
                If ``True``, assume inputs will be unnormalized logits. Otherwise, assume inputs
                are normalized probabilities.

            eps:
                Numerical stabilizer for non-logit inputs
        """
        if from_logits:
            p = x.sigmoid()
            log_p = F.logsigmoid(x)
            log_p_inv = F.logsigmoid(x.neg())
        else:
            p = x
            log_p = x.log().clamp_min(-eps)
            log_p_inv = (1 - x).log().clamp_min(-eps)

        # 2-class normalization divisor
        with torch.no_grad():
            divisor = p.new_tensor(2).log_()

        p_inv = 1 - p
        return log_p.mul(p).add(log_p_inv.mul(p_inv)).neg().div(divisor)

    @staticmethod
    def compute_categorical_entropy(
        x: Tensor,
        dim: int = -1,
        from_logits: bool = True,
        eps: float = 1e6,
    ) -> Tensor:
        r"""Computes normalized categorical entropy along a given tensor dimension. The output is
        dimensionless (i.e. not in nats or bits) because the output is normalized.

        Args:
            x:
                Tensor to compute entropy over

            dim:
                Dimension to compute entropy over. Only relevant for categorical entropy.

            from_logits:
                If ``True``, assume inputs will be unnormalized logits. Otherwise, assume inputs
                are normalized probabilities.

            eps:
                Numerical stabilizer for non-logit inputs
        """
        if from_logits:
            p = x.softmax(dim=dim)
            log_p = F.log_softmax(x, dim=dim)
        else:
            p = x
            log_p = x.log().clamp_min(-eps)

        C = x.shape[dim]
        assert C > 0

        # C-class normalization divisor
        with torch.no_grad():
            divisor = p.new_tensor(C).log_()

        return log_p.mul(p).sum(dim=dim).neg().div(divisor)
