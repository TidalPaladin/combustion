#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from abc import ABC, abstractmethod
from enum import IntEnum, Enum

from torch import Tensor
from typing import Any, Callable, Optional, Tuple, List, Type
from math import sqrt
from functools import partial

class SqueezeExcite(nn.Module):

    def __init__(self, d: int, d_hidden: int, dropout: float = 0):
        super().__init__()
        self.se = nn.Sequential(
            nn.Linear(d, d_hidden),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(d_hidden, d),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        L, N, D = x.shape
        pool = x.mean(dim=0, keepdim=True)
        weight = self.se(pool)
        return x * weight


class FourierMixer(nn.Module):

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

        #if self.nhead != 1:
        #    x = x.view(L, -1, D_head).contiguous()

        x = self._forward(x)

        #if self.nhead != 1:
        #    x = x.view(-1, N, D).contiguous()

        if x.is_complex() and self.real_out:
            x = x.real

        if orig_dtype != x.dtype and not x.is_complex():
            x = x.to(dtype=orig_dtype)

        return x

    def _forward(self, x: Tensor) -> Tensor:
        L_dim, D_dim = 0, -1
        x = torch.fft.fft(x, dim=D_dim, norm=self.norm)
        x = torch.fft.fft(x, dim=L_dim, norm=self.norm)
        #x = torch.fft.fft2(x, dim=(D_dim, L_dim), norm=self.norm).contiguous()
        #x = torch.fft.fft(x, dim=L_dim, norm=self.norm).contiguous()
        return x


class FourierDownsample(FourierMixer):

    def _forward(self, x: Tensor) -> Tensor:
        L_dim, D_dim = 0, -1
        L, D = x.shape[L_dim], x.shape[D_dim]
        x = torch.fft.rfft2(x, dim=(D_dim, L_dim), norm=self.norm)[:L//2].contiguous()
        return x


class FourierUpsample(FourierMixer):

    def _forward(self, x: Tensor) -> Tensor:
        L_dim, D_dim = 0, -1
        x = torch.fft.irfft2(x, dim=(D_dim, L_dim), norm=self.norm).contiguous()
        return x


class FNet(nn.Module):

    def __init__(self, d: int, dim_ff: int, nhead: int = 1, dout: Optional[int] = None, dropout: float = 0.1, norm: str = "ortho", act: nn.Module = nn.SiLU(), use_bn: bool = False):
        super().__init__()
        self.d = d
        self.use_bn = use_bn
        self.d_out = dout or d
        assert self.d % nhead == 0
        if use_bn:
            dropout = 0

        self.mixer = nn.Sequential(
            nn.Linear(d, d),
            FourierMixer(nhead, norm),
            nn.Linear(d, d),
        )
        self.se = SqueezeExcite(d, d // 2)
        self.norm1 = nn.BatchNorm1d(d) if use_bn else nn.LayerNorm(d)
        self.feedforward = nn.Sequential(
            nn.Linear(d, dim_ff),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, self.d_out),
            nn.Dropout(dropout),
            act
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

class FNetDownBlock(nn.Module):

    def __init__(self, d: int, dim_ff: int, repeats: int = 1, nhead: int = 1, dout: Optional[int] = None, dropout: float = 0.1, norm: str = "ortho", act: nn.Module = nn.SiLU(), use_bn: bool = False):
        super().__init__()
        self.blocks = nn.Sequential(*[
            FNet(d, dim_ff, nhead, d, dropout, norm, act, use_bn)
            for _ in range(repeats -1)
        ])
        self.pre_downsample = FNet(d, dim_ff, nhead, d*2, dropout, norm, act, use_bn)
        self.downsample = FourierDownsample(nhead, norm)

    def forward(self, x: Tensor) -> Tensor:
        x = self.blocks(x)
        x = self.pre_downsample(x)
        return self.downsample(x)

class FNetUpBlock(nn.Module):

    def __init__(self, d: int, dim_ff: int, repeats: int = 1, nhead: int = 1, dout: Optional[int] = None, dropout: float = 0.1, norm: str = "ortho", act: nn.Module = nn.SiLU()):
        super().__init__()
        self.blocks = nn.Sequential(*[
            FNet(d, dim_ff, nhead, d, dropout, norm, act)
            for _ in range(repeats-1)
        ])
        self.pre_upsample = FNet(d, dim_ff, nhead, d // 2, dropout, norm, act,)
        self.upsample = FourierUpsample(nhead, norm)

    def forward(self, x: Tensor, skip_conn: Tensor) -> Tensor:
        x = self.blocks(x)
        x = self.pre_upsample(x)
        x = self.upsample(x)
        return x + skip_conn

class WeightedFourierMixer(FourierMixer):

    def _forward(self, x: Tensor) -> Tensor:
        L_dim, D_dim = 0, -1
        weight = torch.fft.fft(x, dim=L_dim, norm=self.norm).contiguous().real
        weight = weight.softmax(dim=L_dim)
        x = x * weight
        return x

class SortedFourierMixer(FourierMixer):

    def __init__(self, d: int, num_freqs: int, norm: str = "ortho", real_out: bool = True):
        super().__init__(1, norm, real_out)
        self.num_freqs = num_freqs
        self.out_size = self.num_freqs // 2 + 1
        D_f = self.num_freqs

        #self.in_proj = nn.Linear(d, 3*d)
        self.query_proj = nn.Linear(d, d)
        self.key_proj = nn.Linear(d, d)
        self.value_proj = nn.Linear(d, d)

        self.spatial = nn.Linear(self.out_size, self.num_freqs)
        
        self.query_router = nn.Sequential(
            nn.Linear(d, D_f),
        )
        self.key_router = nn.Sequential(
            nn.Linear(d, D_f),
        )
        self.value_router = nn.Sequential(
            nn.Linear(d, D_f),
            nn.Softmax(dim=-1)
        )
        self.out_proj = nn.Linear(d, d)

    def forward(self, x: Tensor) -> Tensor:
        L, N, D = x.shape
        W = self.num_freqs
        S = self.out_size
        L_dim, D_dim = 0, -1
        orig_x = x

        #Q = self.query_proj(x)
        #K = self.key_proj(x)
        #V = self.value_proj(x)
        #Q, K, V = self.in_proj(x).chunk(3, dim=-1)

        Q = x

        Qf = self.query_router(Q).swapdims(0, 1).div(L)
        #Kf = self.key_router(K).swapdims(0, 1).div(L)
        #Vf = self.value_router(V).swapdims(0, 1)
        assert Qf.shape == (N, L, W)
        #assert Kf.shape == (N, L, W)
        #assert Vf.shape == (N, L, W)

        Q_routed = Q.movedim(0, -1).bmm(Qf)
        #K_routed = K.movedim(0, -1).bmm(Kf)
        #V_routed = V.movedim(0, -1).bmm(Vf.div(L))
        assert Q_routed.shape == (N, D, W)
        #assert K_routed.shape == (N, D, W)
        #assert V_routed.shape == (N, D, W)

        Q_routed = torch.fft.fft2(Q_routed.float(), dim=(-2, -1), norm="ortho")
        #K_routed = torch.fft.fft2(K_routed.float(), dim=(-2, -1), norm="ortho")
        #V_routed = torch.fft.fft2(V_routed.float(), dim=(-2, -1), norm="ortho")
        attn = Q_routed 
        assert attn.shape == (N, D, W)

        Qf = torch.fft.fft(Qf.float(), dim=(-1), norm="ortho")
        #assert Vf.shape == (N, L, W)

        out = attn.bmm(Qf.swapdims(-1, -2)).movedim(-1, 0)
        assert out.shape == (L, N, D)

        out = out.real.type_as(x)
        return self.out_proj(out)





class FourierTransformer(FNet):

    def __init__(self, d: int, dim_ff: int, nhead: int = 1, dout: Optional[int] = None, dropout: float = 0.1, norm: str = "ortho", act: nn.Module = nn.SiLU(), use_bn: bool = False):
        super().__init__(d, dim_ff, nhead, dout, dropout, norm, act, use_bn)
        del(self.mixer)
        self.norm = norm
        self.mixer = SortedFourierMixer(d, 128)

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm1(x + self.mixer(x))
        x = self.feedforward(x)
        if self.d_out == self.d:
            x = x + self.feedforward(x)
        else:
            x = self.feedforward(x)
        out = self.norm2(x)
        return out
