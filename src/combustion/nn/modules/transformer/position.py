#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import math
from copy import deepcopy
from torch import Tensor
from typing import Iterable, Optional, Dict, Any
from .common import SequenceBatchNorm, SequenceInstanceNorm

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
                pe[pos, i] = \
                math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = \
                math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
                
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
 
    
    def forward(self, x: Tensor) -> Tensor:
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        #add constant to embedding
        seq_len = x.size(1)
        x = x + self.pe[:,:seq_len]
        return x

class RelativePositionalEncoder(nn.Module):
    def __init__(self, num_coords: int, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.num_coords = num_coords
        self.linear = nn.Sequential(
            nn.Linear(num_coords, d_model),
            nn.ReLU()
        )
    
    def forward(self, center: Tensor, points: Tensor) -> Tensor:
        L, N, C = center.shape
        K, L, N, C = points.shape
        assert C == self.num_coords
        with torch.no_grad():
            delta = points - center.view(1, L, N, C)
        return self.linear(delta)


class LearnableFourierFeatures(nn.Module):
    def __init__(
        self, 
        d_in: int, 
        num_features: int, 
        d_out: int, 
        d_hidden: Optional[int] = None,
        dropout: float = 0.0,
        sigma: Optional[float] = None, 
        init: str = "normal",
        act: nn.Module = nn.ReLU(),
        **kwargs
    ):
        super().__init__()
        self.num_features = num_features
        self.features = nn.Linear(d_in, num_features // 2, bias=False)

        d_hidden = d_hidden or d_out
        self.mlp = nn.Sequential(
            nn.Linear(num_features, d_hidden),
            deepcopy(act),
            nn.Linear(d_hidden, d_out),
            deepcopy(act),
            nn.Dropout(dropout),
        )
        self.norm = SequenceBatchNorm(d_out)

        #sigma = sigma or num_features / (2 * math.pi) ** 0.5
        sigma = sigma or 1.0
        if init == "ortho":
            torch.nn.init.orthogonal_(self.features.weight, sigma)
        elif init == "normal":
            torch.nn.init.normal_(self.features.weight, 0, sigma)
        else:
            raise ValueError(init)

    def weight_decay_dict(self, val: float) -> Dict[str, Any]:
        return {"params": self.features.parameters(), "weight_decay": val}

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = torch.cat([x.cos(), x.sin()], dim=-1)
        x = x / self.num_features ** 0.5
        x = self.mlp(x)
        x = self.norm(x)
        return x

    @staticmethod
    def from_grid(dims: Iterable[int], proto: Optional[Tensor] = None, requires_grad: bool = True, **kwargs) -> Tensor:
        if proto is not None:
            device = proto.device
            dtype = proto.dtype
            _kwargs = {"device": device, "dtype": dtype}
            _kwargs.update(kwargs)
            kwargs = _kwargs

        with torch.no_grad():
            lens = tuple(torch.arange(d, **kwargs) for d in dims)
            grid = torch.stack(torch.meshgrid(*lens), dim=0)
            grid = grid.movedim(0, -1).view(-1, 1, len(lens))

        requires_grad = requires_grad or (proto is not None and proto.requires_grad)
        grid.requires_grad = requires_grad
        return grid


class RelativeLearnableFourierFeatures(LearnableFourierFeatures):

    def forward(self, center: Tensor, neighbors: Tensor) -> Tensor:
        L, N, C = center.shape
        K, L, N, C = neighbors.shape
        delta = neighbors - center.view(1, L, N, C)
        return super().forward(delta)


class FourierLinear(nn.Module):
    def __init__(
        self, 
        d_in: int, 
        num_features: int, 
        d_out: int, 
        sigma: Optional[float] = None, 
        init: Optional[str] = None,
    ):
        super().__init__()
        init = init or ("normal" if d_out != d_in else "ortho")
        self.num_features = num_features
        self.features = nn.Linear(d_in, num_features, bias=False)
        self.out = nn.Linear(2*num_features, d_out)

        #sigma = sigma or num_features / (2 * math.pi) ** 0.5
        sigma = sigma or 1.0
        if init == "ortho":
            torch.nn.init.orthogonal_(self.features.weight, sigma)
        elif init == "normal":
            torch.nn.init.normal_(self.features.weight, 0, sigma)
        else:
            raise ValueError(init)

    def weight_decay_dict(self, val: float) -> Dict[str, Any]:
        return {"params": self.features.parameters(), "weight_decay": val}

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x) * (2 * math.pi)
        x = torch.cat([x.cos(), x.sin()], dim=-1)
        x = x / self.num_features ** 0.5
        x = self.out(x)
        return x

    @staticmethod
    def use_fourier(module: nn.Module):
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                layer = FourierLinear(child.in_features, child.in_features, child.out_features)
                setattr(module, name, layer)
            elif not isinstance(child, (LearnableFourierFeatures, nn.MultiheadAttention, FourierLinear)):
                FourierLinear.use_fourier(child)
