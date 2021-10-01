#!/usr/bin/env python
# -*- coding: utf-8 -*-

from copy import deepcopy
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from .common import MLP


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
    ):
        super().__init__()
        self.d = d
        self.d_out = dout or d
        assert self.d % nhead == 0

        # NOTE:
        #    1. Using norm other than "ortho" seems to reduce performance
        #    2. Adding LayerNorm seems to reduce performance
        self.mixer = FourierMixer(nhead, norm)
        self.norm1 = nn.LayerNorm(d)
        self.feedforward = MLP(d, dim_ff, self.d_out, act=act, dropout=dropout)
        self.norm2 = nn.LayerNorm(self.d_out)

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm1(x + self.mixer(x))
        x = self.norm2(x + self.feedforward(x))
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
