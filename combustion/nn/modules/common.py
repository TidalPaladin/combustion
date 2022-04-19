#!/usr/bin/env python
# -*- coding: utf-8 -*-

from copy import deepcopy
from math import log, pi
from typing import Any, Iterable, Optional

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

    def forward(self, x: Tensor, original: Optional[Tensor] = None) -> Tensor:
        if not self.training or self.ratio == 1:
            return x
        mask = self.get_mask(x)
        dropped = x * mask
        if original is not None:
            return original * (~mask) + dropped
        else:
            return dropped


class MLP(nn.Module):
    def __init__(
        self,
        d: int,
        d_hidden: int,
        d_out: Optional[int] = None,
        dropout: float = 0,
        act: nn.Module = nn.ReLU(),
        se_ratio: Optional[int] = None,
        num_hidden_layers: int = 1,
        norm_layer: Any = nn.Identity,
    ):
        super().__init__()
        self.d_out = d_out = d_out or d
        self.d_in = d

        self.input = nn.Sequential(
            nn.Linear(d, d_hidden),
            norm_layer(d_hidden),
            nn.Dropout(dropout),
            deepcopy(act),
        )

        self.hidden_layers = nn.ModuleList()
        for _ in range(num_hidden_layers - 1):
            hidden_layer = nn.Sequential(
                nn.Linear(d_hidden, d_hidden),
                norm_layer(d_hidden),
                nn.Dropout(dropout),
                deepcopy(act),
            )
            self.hidden_layers.append(hidden_layer)

        self.output = nn.Linear(d_hidden, d_out)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)
            else:
                nn.init.constant_(p, 0)

    def init_binary_prior(self, p: float, eps: Optional[float] = None) -> None:
        logit_bias = float(torch.tensor(p).logit(eps).item())
        nn.init.constant_(self.output.bias, logit_bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.input(x)
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
        x = self.output(x)
        return x


class SqueezeExcite(nn.Module):
    def __init__(self, d_in, d_squeeze, act: nn.Module = nn.ReLU(), final_act: nn.Module = nn.Mish()):
        super().__init__()
        self.d_in = d_in
        self.se = nn.Sequential(
            nn.Linear(d_in, d_squeeze), deepcopy(act), nn.Linear(d_squeeze, d_in), deepcopy(final_act)
        )

    def forward(self, x: Tensor) -> Tensor:
        weights = self.se(x.mean(dim=0, keepdim=True))
        return x * weights


class FourierEncoder(nn.Module):
    def __init__(self, max_freq: float, num_bands: int, base: int = 2, out_dim: int = -1):
        super().__init__()
        self.num_bands = num_bands
        self.max_freq = max_freq
        self.out_dim = out_dim
        upper_bound = log(max_freq / 2) / log(base)
        self.register_buffer("scales", torch.logspace(1.0, upper_bound, num_bands, base=base))

    def extra_repr(self) -> str:
        return f"max_freq={self.max_freq}, num_bands={self.num_bands}"

    @property
    def output_channels(self) -> int:
        return self.num_bands * 2

    def forward(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(-1)
        x = x * self.scales * pi
        x = torch.cat([x.sin(), x.cos()], dim=-1)
        if self.out_dim != -1:
            x = x.movedim(-1, self.out_dim)
        return x

    @staticmethod
    def from_grid(
        dims: Iterable[int],
        proto: Optional[Tensor] = None,
        requires_grad: bool = True,
        normalize: bool = True,
        **kwargs,
    ) -> Tensor:
        if proto is not None:
            device = proto.device
            dtype = proto.dtype
            _kwargs = {"device": device, "dtype": dtype}
            _kwargs.update(kwargs)
            kwargs = _kwargs

        with torch.no_grad():
            prod = int(torch.tensor(dims).prod().item())
            grid = torch.arange(prod, **kwargs)

            if normalize:
                C = grid.shape[0]
                grid = grid.float()
                scale = grid.view(C, -1).amax(dim=-1).view(C, *((1,) * (grid.ndim - 1)))
                grid.div_(scale).sub_(0.5).mul_(2)

            grid = grid.movedim(0, -1).view(-1, 1, *dims)

        requires_grad = requires_grad or (proto is not None and proto.requires_grad)
        grid.requires_grad = requires_grad
        return grid
