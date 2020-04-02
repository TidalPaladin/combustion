#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
from ignite.engine import Engine
from torch import Tensor


class SupervisedEvalFunc:
    def __init__(self, model, device=None):
        # type: (nn.Module, Optional[Any])
        if not isinstance(model, nn.Module):
            raise TypeError(f"model must be a nn.Module, found {type(model)}")
        self.model = model
        self.device = device

    def step(self, engine, batch):
        # type: (Engine, Any) -> Tuple[Tensor, Tensor]
        raise NotImplementedError(f"must implement step method")

    def __call__(self, engine, batch):
        # type: (Engine, Any) -> float
        self.model.eval()

        if self.device is not None:
            batch = batch.cuda(self.device, non_blocking=True)
            engine.state.batch = batch

        # forward pass, loss, backward pass
        y_pred, y_true = self.step(engine, batch)
        return y_pred, y_true

    def __repr__(self):
        name = self.__class__.__name__
        model_name = self.model.__class__.__name__
        s = f"{name}({model_name})"
        return s
