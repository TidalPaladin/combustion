#!/usr/bin/env python
# -*- coding: utf-8 -*-
from argparse import Namespace
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
from ignite.engine import Engine
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer


try:
    import apex
    from apex import amp
except ImportError:
    apex = None
    amp = None


class SupervisedTrainFunc:
    r"""
    Base class for training functions that can be called by Ignite engines.

    Args:
        model (nn.Module): Model to load
        optimizer (Optimizer): Optimizer for training
        criterion (nn.Module): Loss function for training
        grad_steps (int, optional): Iterations between callign `optimizer.step()`
        grad_norm (float, optional): If given, threshold for L2 graident clipping on model weights
        out_grad_norm (float, optional): If given, threshold for L2 graident clipping on outputs
        amp (bool, optional): If true, use AMP from the Apex library
    """

    def __init__(
        self,
        model,
        optimizer,
        criterion,
        device=None,
        grad_steps=1,
        grad_norm=None,
        out_grad_norm=None,
        amp=False,
        schedule=None,
    ):
        # type: (nn.Module, Optimizer, nn.Module, Optional[Any], int, Optional[float], Optional[float], bool)
        if not isinstance(model, nn.Module):
            raise TypeError(f"model must be a nn.Module, found {type(model)}")
        if not isinstance(optimizer, Optimizer):
            raise TypeError(f"optimizer must be an optim.Optimizer, found {type(optimizer)}")
        if not isinstance(criterion, nn.Module):
            raise TypeError(f"optimizer must be an optim.Optimizer, found {type(optimizer)}")
        self.amp = amp

        self.grad_steps = int(grad_steps)
        if self.grad_steps <= 0:
            raise ValueError(f"grad_steps must be >=1, got {self.grad_steps}")

        if grad_norm is not None:
            self.grad_norm = float(grad_norm)
            if self.grad_norm <= 0:
                raise ValueError(f"grad_norm must be >0, got {self.grad_norm}")
        else:
            self.grad_norm = None

        if out_grad_norm is not None:
            self.out_grad_norm = float(out_grad_norm)
            if self.out_grad_norm <= 0:
                raise ValueError(f"out_grad_norm must be >0, got {self.out_grad_norm}")
        else:
            self.out_grad_norm = None

        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion
        self.lr_schedule = schedule

    def step(self, engine, batch):
        # type: (Engine, Any) -> Tuple[Tensor, Tensor]
        raise NotImplementedError(f"must implement step method")

    def __call__(self, engine, batch):
        # type: (Engine, Any) -> float

        # zero grads at start
        if engine.state.iteration == 1:
            self.optimizer.zero_grad()
        self.model.train()

        if self.device is not None:
            batch = batch.cuda(self.device, non_blocking=True)
            engine.state.batch = batch

        # forward pass, loss, backward pass
        y_pred, y_true = self.step(engine, batch)
        loss = self.criterion(y_pred, y_true)
        self.backward(loss, engine.state.iteration)

        # grad clipping if requested
        self._maybe_clip_output_grads(y_pred)
        self._maybe_clip_model_grads()

        return loss.item()

    def __repr__(self):
        # type: () -> str
        name = self.__class__.__name__
        model_name = self.model.__class__.__name__
        opt_name = self.optimizer.__class__.__name__
        criterion_name = self.criterion.__class__.__name__
        s = f"{name}({model_name}, {opt_name}, {criterion_name}"
        if self.grad_steps != 1:
            s += ", grad_steps={self.grad_steps}"
        if self.grad_norm is not None:
            s += ", grad_norm={self.grad_norm}"
        if self.out_grad_norm is not None:
            s += ", out_grad_norm={self.out_grad_norm}"
        return s + ")"

    def backward(self, loss, step):
        # type: (Tensor, int)
        """backward pass with optimizer step every x iterations"""
        if self.amp:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        # optimizer step every `self.grad_steps` iterations
        if step % self.grad_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

    @classmethod
    def from_args(cls, args, model: nn.Module, optimizer, criterion, device=None, lr_schedule=None):
        # type: (Namespace, nn.Module, Optimizer, nn.Module, Optional[Any]) -> SupervisedTrainFunc
        amp = args.opt_level is not None
        return cls(
            model, optimizer, criterion, device, args.grad_steps, args.grad_clip, args.out_grad_clip, amp, lr_schedule
        )

    def _maybe_clip_model_grads(self):
        if self.grad_norm is not None:
            clip_grad_norm_(self.model.parameters(), self.grad_norm)

    def _maybe_clip_output_grads(self, outputs: Tensor):
        if self.out_grad_norm is not None:
            clip_grad_norm_(outputs, self.out_grad_norm)
