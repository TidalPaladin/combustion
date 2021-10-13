#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
from typing import Iterable, Optional, Type

import torch
import torch.nn as nn
from torch import Tensor

from .common import MLP


class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.emb = nn.Embedding(max_seq_len, dim)

    def forward(self, x: Tensor) -> Tensor:
        L, N, E = x.shape
        assert L <= self.max_seq_len
        t = torch.arange(L, device=x.device)
        emb = self.emb(t).view(L, 1, E)
        return x + emb


class PositionalEncoder(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int = 80):
        super().__init__()
        self.d_model = d_model

        # create constant 'pe' matrix with values dependant on
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        # add constant to embedding
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return x


class RelativePositionalEncoder(nn.Module):
    def __init__(self, num_coords: int, d_out: int, **kwargs):
        super().__init__()
        self.mlp = MLP(num_coords, d_out, d_out, **kwargs)
        self.norm = nn.LayerNorm(d_out)
        self.d_out = d_out
        self.num_coords = num_coords

    def forward(self, center: Tensor, points: Tensor) -> Tensor:
        L, N, C = center.shape
        K, L, N, C = points.shape
        assert C == self.num_coords
        with torch.no_grad():
            delta = points - center.view(1, L, N, C)
        out = self.norm(self.mlp(delta))
        return out


class LearnableFourierFeatures(nn.Module):
    r"""Implements Learnable Fourier Features as described in `Learnable Fourier Features`_. LFF is a
    positional encoding mechanism based on random fourier features.

    Args:
        d_in:
            Number of input channels

        num_features:
            Number of random features

        d_out:
            Number of output channels

        d_hidden:
            Hidden dimension size of the MLP

        dropout:
            Dropout rate for the MLP

        gamma:
            Initializer value for the random features. Random features are sampled from
            :math:`\mathrel{N}(0, \gamma^{-2})`.

        act:
            Activation function for the MLP

        norm_layer:
            Normalization layer to use. Defaults to batch norm.

    Keyword Args:

        Forwarded to initialization of ``norm_layer``.

    Shapes:
        * ``x`` - :math:`(L, N, D_{in})`
        * Output - :math:`(L, N, D_{out})`

    .. _Learnable Fourier Features: https://arxiv.org/abs/2106.02795v2
    """

    def __init__(
        self,
        d_in: int,
        num_features: int,
        d_out: int,
        d_hidden: Optional[int] = None,
        dropout: float = 0.0,
        gamma: float = 1.0,
        act: nn.Module = nn.ReLU(),
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        **kwargs,
    ):
        super().__init__()
        self.num_features = num_features
        self.features = nn.Linear(d_in, num_features // 2, bias=False)

        d_hidden = d_hidden or d_out
        self.mlp = MLP(num_features, d_hidden, d_out, dropout=dropout, act=act)
        self.norm = norm_layer(d_out, **kwargs)  # type: ignore
        torch.nn.init.normal_(self.features.weight, 0, gamma ** -2.0)

    @property
    def trainable(self) -> bool:
        return self.features.weight.requires_grad

    @trainable.setter
    def trainable(self, state: bool):
        self.features.weight.requires_grad = state

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = torch.cat([x.cos(), x.sin()], dim=-1)
        x = x / self.num_features ** 0.5
        x = self.mlp(x)
        x = self.norm(x)
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
            lens = tuple(torch.arange(d, **kwargs) for d in dims)
            grid = torch.stack(torch.meshgrid(*lens), dim=0)

            if normalize:
                C = grid.shape[0]
                grid = grid.float()
                scale = grid.view(C, -1).amax(dim=-1).view(C, *((1,) * (grid.ndim - 1)))
                grid.div_(scale).sub_(0.5).mul_(2)

            grid = grid.movedim(0, -1).view(-1, 1, len(lens))

        requires_grad = requires_grad or (proto is not None and proto.requires_grad)
        grid.requires_grad = requires_grad
        return grid

    # def visualize(self, grid: Tensor, mlp: bool = True):
    #    try:
    #        import matplotlib.pyplot as plt
    #    except Exception:
    #        print("Visualization requires matploitlib")
    #        raise

    #    if grid.ndim > 3:
    #        raise ValueError(f"Invalid ndim {grid.ndim}")

    #    fig = plt.figure()
    #    rows, cols = 2, 3

    #    plt.subplot(2, 3, 1)
    #    plt.plot(t, s1)
    #    plt.subplot(212)
    #    plt.plot(t, 2*s1)


class RelativeLearnableFourierFeatures(LearnableFourierFeatures):
    r"""Subclass of :class:`LearnableFourierFeatures` that builds positional encodings
    for a set of local neighborhoods.

    Shapes:
        * ``center`` - :math:`(L, N, D_{in})`
        * ``neighbors`` - :math:`(K, L, N, D_{in})` where :math:`K` is the neighborhood size
        * Output - :math:`(L, N, D_{out})`
    """

    def forward(self, center: Tensor, neighbors: Tensor) -> Tensor:
        L, N, C = center.shape
        K, L, N, C = neighbors.shape
        delta = neighbors - center.view(1, L, N, C)
        return super().forward(delta)


class FourierLogspace(nn.Module):
    scales: Tensor

    def __init__(
        self, 
        d_in: int, 
        d_out: int, 
        max_freq: float, 
        num_bands: int, 
        zero_one_norm: bool = True,
        base: int = 2,
        **kwargs
    ):
        super().__init__()
        start = 0
        stop = math.log(max_freq / 2) / math.log(2)
        self.register_buffer("scales", math.pi * torch.logspace(start, stop, num_bands, base=base))
        d_mlp = self.scales.numel() * d_in * 2
        dim_ff = max(d_mlp, d_in)
        self.mlp = MLP(d_mlp, dim_ff, d_out, **kwargs)

    @property
    def num_bands(self) -> int:
        return self.scales.numel()

    def forward(self, x: Tensor) -> Tensor:
        L, N, C = x.shape
        x = x.unsqueeze(-1)
        x = x * self.scales
        x = torch.cat([x.sin(), x.cos()], dim=-1)
        x = x.view(L, N, -1)
        x = self.mlp(x)
        return x


class RelativeFourierLogspace(FourierLogspace):
    def forward(self, center: Tensor, neighbors: Tensor) -> Tensor:
        L, N, C = center.shape
        K, L, N, C = neighbors.shape
        delta = neighbors - center.view(1, L, N, C)
        return super().forward(delta.view(-1, N, C)).view(K, L, N, C)
