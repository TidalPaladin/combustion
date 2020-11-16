#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Callable, Optional

import torch
from pytorch_lightning.metrics import Metric
from pytorch_lightning.metrics.functional import auroc
from torch import Tensor


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
