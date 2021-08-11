#!/usr/bin/env python
# -*- coding: utf-8 -*-

from copy import deepcopy
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from .common import MLP, SqueezeExcite


class FourierMixer(nn.Module):
    r"""FNet Mixer """

    def __init__(self, nhead: int = 1, norm: str = "ortho", real_out: bool = True):
        super().__init__()
        self.norm = norm
        self.nhead = nhead
        self.real_out = real_out

    def extra_repr(self) -> str:
        s = f"nhead={self.nhead}, norm={self.norm}"
        return s

    def forward(self, x: Tensor) -> Tensor:
        orig_dtype = x.dtype
        if x.dtype == torch.half:
            x = x.float()

        L, N, D = x.shape
        L_dim, D_dim = 0, -1
        D_head = D // self.nhead
        assert D_head * self.nhead == D

        x = self._forward(x)

        if x.is_complex() and self.real_out:
            x = x.real

        if orig_dtype != x.dtype and not x.is_complex():
            x = x.to(dtype=orig_dtype)

        return x

    def _forward(self, x: Tensor) -> Tensor:
        L_dim, D_dim = 0, -1
        x = torch.fft.fft(x, dim=D_dim, norm="ortho")
        x = torch.fft.fft(x, dim=L_dim, norm="ortho")
        return x


class FNet(nn.Module):
    def __init__(
        self,
        d: int,
        dim_ff: int,
        nhead: int = 1,
        dout: Optional[int] = None,
        dropout: float = 0.0,
        norm: str = "ortho",
        act: nn.Module = nn.SiLU(),
        use_bn: bool = False,
    ):
        super().__init__()
        self.d = d
        self.use_bn = use_bn
        self.d_out = dout or d
        assert self.d % nhead == 0
        if use_bn:
            dropout = 0

        # NOTE:
        #    1. Using norm other than "ortho" seems to reduce performance
        #    2. Adding LayerNorm seems to reduce performance
        self.mixer = nn.Sequential(
            nn.Linear(d, d),
            FourierMixer(nhead, norm),
            nn.Linear(d, d),
        )

        self.se = SqueezeExcite(d, d // 2)
        self.norm1 = nn.BatchNorm1d(d) if use_bn else nn.LayerNorm(d)
        self.feedforward = nn.Sequential(
            nn.Linear(d, dim_ff), nn.Dropout(dropout), nn.Linear(dim_ff, self.d_out), nn.Dropout(dropout), act
        )
        self.norm2 = nn.BatchNorm1d(self.d_out) if use_bn else nn.LayerNorm(self.d_out)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.mixer(x)

        if self.use_bn:
            x = self.norm1(x.permute(1, 2, 0)).permute(2, 0, 1)
        else:
            x = self.norm1(x)

        if self.d_out == self.d:
            x = x + self.feedforward(x)
        else:
            x = self.feedforward(x)

        x = x + self.se(x)

        if self.use_bn:
            x = self.norm2(x.permute(1, 2, 0)).permute(2, 0, 1)
        else:
            x = self.norm2(x)

        return x


class FourierTransformer(nn.Module):
    def __init__(self, d: int, d_mid: int, dropout: float = 0.1, act: nn.Module = nn.SiLU()):
        super().__init__()
        self.proj_fft = nn.Linear(d, d_mid)
        self.weight = nn.Linear(d, d_mid)

        self.linear_r = nn.Linear(d_mid, d_mid)
        self.norm_r = nn.LayerNorm(d_mid)

        self.linear_i = nn.Linear(d_mid, d_mid)
        self.norm_i = nn.LayerNorm(d_mid)

        self.mlp = MLP(d_mid, d_mid, d, dropout=dropout, act=deepcopy(act))
        self.norm = nn.LayerNorm(d)

    def forward(self, x: Tensor) -> Tensor:
        orig_x = x
        f = self.proj_fft(x)
        weight = self.weight(x).softmax(dim=0)

        with torch.cuda.amp.autocast(enabled=False):
            f = torch.fft.fft(f.float(), dim=-1, norm="ortho")

        i = f.imag.type_as(x)
        r = f.real.type_as(x)
        i = self.norm_i(self.linear_i(i))
        r = self.norm_r(self.linear_r(r))

        itot = i.sum(dim=0, keepdim=True)
        rtot = r.sum(dim=0, keepdim=True)

        r = r + weight * rtot
        i = i + weight * itot

        with torch.cuda.amp.autocast(enabled=False):
            f = torch.complex(r.float(), i.float())
            x = torch.fft.ifft(f.float(), norm="ortho").real
        x = x.type_as(orig_x)

        return self.norm(orig_x + self.mlp(x))
