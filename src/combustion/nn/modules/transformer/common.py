#!/usr/bin/env python
# -*- coding: utf-8 -*-

from copy import deepcopy
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class SequenceBatchNorm(nn.BatchNorm1d):
    r"""Batch norm for sequences of shape :math:`(L, N, C)`"""

    def forward(self, x: Tensor) -> Tensor:
        orig_x = x
        N, D = x.shape[-2:]
        x = x.view(-1, N, D)
        x = x.movedim(0, -1).contiguous()
        x = super().forward(x)
        x = x.movedim(-1, 0).contiguous()
        x = x.view_as(orig_x).contiguous()
        return x


class SequenceInstanceNorm(nn.InstanceNorm1d):
    r"""Instance norm for sequences of shape :math:`(L, N, C)`"""

    def forward(self, x: Tensor) -> Tensor:
        orig_x = x
        N, D = x.shape[-2:]
        x = x.view(-1, N, D)
        x = x.movedim(0, -1).contiguous()
        x = super().forward(x)
        x = x.movedim(-1, 0).contiguous()
        x = x.view_as(orig_x).contiguous()
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

    @staticmethod
    def use_instancenorm(module: nn.Module, **kwargs):
        for name, layer in module.named_children():
            if hasattr(layer, "dropout") and isinstance(layer.dropout, float):
                layer.dropout = 0
            if isinstance(layer, nn.LayerNorm):
                d = layer.normalized_shape[0]
                new_layer = SequenceInstanceNorm(d, **kwargs)
                setattr(module, name, new_layer)
            elif isinstance(layer, nn.Dropout):
                new_layer = nn.Identity()
                setattr(module, name, new_layer)
            else:
                BatchNormMixin.use_instancenorm(layer)

    @staticmethod
    def layernorm_nonaffine(module: nn.Module, **kwargs):
        for name, layer in module.named_children():
            if isinstance(layer, nn.LayerNorm):
                d = layer.normalized_shape[0]
                new_layer = nn.LayerNorm(d, elementwise_affine=False, **kwargs)
                setattr(module, name, new_layer)
            else:
                BatchNormMixin.layernorm_nonaffine(layer)


class DropPath(nn.Module):
    r"""DropPath, otherwise known as stochastic depth"""

    def __init__(self, ratio: float):
        super().__init__()
        assert ratio >= 0 and ratio < 1.0
        self.ratio = 1.0 - abs(float(ratio))

    def get_mask(self, x: Tensor) -> Tensor:
        L, N, D = x.shape
        with torch.no_grad():
            mask = torch.rand(N, device=x.device).add_(self.ratio).floor_().bool()
            mask = mask.view(1, N, 1)
        assert mask.ndim == x.ndim
        assert mask.shape[1] == N
        return mask.bool()

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.ratio == 1:
            return x
        return x * self.get_mask(x)


class MLP(nn.Module):
    def __init__(
        self, d: int, d_hidden: int, d_out: Optional[int] = None, dropout: float = 0, act: nn.Module = nn.ReLU()
    ):
        super().__init__()
        d_out = d_out or d
        self.l1 = nn.Linear(d, d_hidden)
        self.d1 = nn.Dropout(dropout)
        self.l2 = nn.Linear(d_hidden, d_out)
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
        self.d_in = d_in
        self.se = nn.Sequential(nn.Linear(d_in, d_squeeze), act, nn.Linear(d_squeeze, d_in), nn.Sigmoid())

    def forward(self, x: Tensor) -> Tensor:
        weights = self.se(x.mean(dim=0, keepdim=True))
        return x + x * weights
