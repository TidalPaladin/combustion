#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import math
from torch import Tensor
from typing import Iterable, Optional

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
        gamma: float = 1, 
        init: str = "normal",
        norm_layer = nn.LayerNorm
    ):
        super().__init__()
        self.num_features = num_features
        self.features = nn.Linear(d_in, num_features, bias=False)
        cat_features = 2 * num_features
        self.mlp = nn.Sequential(
            nn.Linear(cat_features, cat_features),
            nn.ReLU(),
            nn.Linear(cat_features, d_out),
            nn.ReLU()
        )
        self.norm = norm_layer(d_out)

        if init == "ortho":
            torch.nn.init.orthogonal_(self.features.weight, gamma)
        elif init == "normal":
            torch.nn.init.normal_(self.features.weight, 0, gamma)

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x) 
        x = torch.cat([x.cos(), x.sin()], dim=-1)
        x = x / self.num_features ** 0.5
        x = self.mlp(x)

        if isinstance(self.norm, nn.LayerNorm):
            x = self.norm(x)
        else:
            x = self.norm(x.movedim(0, -1)).movedim(-1, 0).contiguous()

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
