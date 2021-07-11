#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from abc import ABC, abstractmethod
from enum import IntEnum, Enum
from dataclasses import dataclass, field

from torch import Tensor
from typing import Any, Callable, Optional, Tuple, List, Type
from math import sqrt
from functools import partial
from copy import deepcopy

class SequenceBatchNorm(nn.BatchNorm1d):

    def forward(self, x: Tensor) -> Tensor:
        x = x.movedim(0, -1).contiguous()
        x = super().forward(x)
        x = x.movedim(-1, 0).contiguous()
        return x


class BatchNormMixin:

    @staticmethod
    def use_batchnorm(module: nn.Module, **kwargs):
        for name, layer in module.named_children():
            if hasattr(layer, "dropout") and isinstance(layer.dropout, float):
                layer.dropout = 0
            if isinstance(layer, nn.LayerNorm):
                d = layer.normalized_shape[0]
                new_layer = SequenceBatchNorm(d, **kwargs)
                setattr(module, name, new_layer)
            elif isinstance(layer, nn.Dropout):
                new_layer = nn.Identity()
                setattr(module, name, new_layer)
            else:
                BatchNormMixin.use_batchnorm(layer)


class DropPath(nn.Module):
    def __init__(self, ratio: float):
        super().__init__()
        self.ratio = 1.0 - abs(float(ratio))
        assert self.ratio >= 0 and self.ratio < 1.0

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or not self.ratio:
            return x

        L, N, D = x.shape
        with torch.no_grad():
            mask = self.ratio + torch.rand(N).type_as(x).floor_()
            mask = mask.view(1, N, 1)

        assert mask.ndim == x.ndim
        assert mask.shape[1] == N
        return x / self.ratio * mask


class MLP(nn.Module):
    def __init__(self, d: int, d_hidden: int, dropout: float = 0, act: nn.Module = nn.ReLU()):
        super().__init__()
        self.l1 = nn.Linear(d, d_hidden)
        self.d1 = nn.Dropout(dropout)
        self.l2 = nn.Linear(d_hidden, d)
        self.d2 = nn.Dropout(dropout)
        self.act = deepcopy(act)

    def forward(self, x: Tensor) -> Tensor:
        x = self.l1(x)
        x = self.act(x)
        x = self.d1(x)
        x = self.l2(x)
        x = self.act(x)
        x = self.d2(x)
        return x


class SqueezeExcite(nn.Module):

    def __init__(self, d_in, d_squeeze, act: nn.Module = nn.ReLU()):
        super().__init__()
        self.se = nn.Sequential(
            nn.Linear(d_in, d_squeeze),
            act,
            nn.Linear(d_squeeze, d_in),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        weights = self.se(x.mean(dim=0, keepdim=True))
        return x + x * weights
