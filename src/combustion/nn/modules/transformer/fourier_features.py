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


def _sample_orth_matrix(size, var: float = 1, device: torch.device = torch.device("cpu")):
  """Samples orthogonal matrix to reduce variance for random features."""
  ss1 = torch.normal(0, var, (size, size), device=device)
  subspace = torch.tril(ss1)
  subspace = subspace / torch.sqrt((subspace**2).sum(0, keepdim=True))

  S = torch.triu(subspace.T.mm(subspace)) - 0.5 * torch.eye(
      subspace.shape[1], device=device)

  result = torch.eye(
      subspace.shape[0], device=device) - subspace.mm(torch.inverse(S)).mm(
          subspace.T)

  return result

def _sample_matrix(rows, cols, var: float = 1, device: torch.device = torch.device("cpu")):
  """Samples orthogonal matrix to reduce variance for random features."""
  return torch.normal(0, var, (rows, cols), device=device)


class RandomFeatures(nn.Module):
    projection_matrix: Tensor
    redraw_step: Tensor

    def __init__(
        self, 
        d: int, 
        num_features: int, 
        num_heads: int = 1,
        trainable: bool = False,
        var: float = 1,
        feature_redraw_interval: Optional[int] = None,
        batch_first: bool = False
    ):
        super().__init__()
        self.d = d
        self.num_heads = num_heads
        self.num_features = num_features
        self.var = var
        self.batch_first = batch_first
        assert feature_redraw_interval is None or not trainable
        self.trainable = trainable
        self.feature_redraw_interval = int(feature_redraw_interval or 0)
        assert num_features % num_heads == 0

        projection_matrix = self.create_projection()
        if self.trainable:
            self.projection_matrix = nn.Parameter(projection_matrix)
        else:
            self.register_buffer('projection_matrix', projection_matrix)
            self.register_buffer("redraw_step", torch.tensor(0))
            self.register_forward_pre_hook(self.__class__._projection_redraw_hook)

    def extra_repr(self) -> str:
        s = ", ".join(
            f"{k}={getattr(self, k)}" for k in ("d", "num_features", "trainable")
        )
        return s

    def forward(self, data: Tensor) -> Tensor:
        if self.batch_first:
            N, L = data.shape[:2]
            out = self.batch_first_forward(data)
        else:
            L, N = data.shape[:2]
            out = self.batch_second_forward(data)
        H = self.num_heads
        R = self.projection_matrix.shape[-1]
        assert out.shape == (N, H, L, R)
        return out

    def batch_first_forward(self, data: Tensor) -> Tensor:
        projection = self.projection_matrix
        H = self.num_heads
        N, L, D = data.shape
        D_head = D // self.num_heads
        _, R = projection.shape

        data = data.view(N, L, H, D_head).swapdims(1, 2)
        projection = projection.view(1, self.num_heads, D_head, -1).expand(N, -1, -1, -1)
        assert projection.shape == (N, H, D_head, R)
        assert data.shape == (N, H, L, D_head)

        proj_data = data.matmul(projection)
        proj_data = proj_data.view(N, H, L, R)
        return proj_data

    def batch_second_forward(self, data: Tensor) -> Tensor:
        projection = self.projection_matrix
        H = self.num_heads
        L, N, D = data.shape
        D_head = D // self.num_heads
        _, R = projection.shape

        data = data.view(L, N, H, D_head).movedim(0, -2)
        projection = projection.view(1, self.num_heads, D_head, -1).expand(N, -1, -1, -1)
        assert data.shape == (N, H, L, D_head)
        assert projection.shape == (N, H, D_head, R)

        proj_data = data.matmul(projection)
        proj_data = proj_data.view(N, H, L, R)
        return proj_data

    @torch.no_grad()
    def create_projection(self, **kwargs) -> Tensor:
        r"""Creates a projection matrix using positive Gaussian orthogonal random features."""
        D, R = self.d, self.num_features
        mat = _sample_matrix(D, R, self.var, **kwargs)
        return mat 

    @torch.no_grad()
    def redraw_projection_matrix(self, **kwargs) -> None:
        r"""Redraws the projection matrix and places it into the buffer"""
        projections = self.create_projection(**kwargs)
        self.projection_matrix.copy_(projections)

    @staticmethod
    def _projection_redraw_hook(module: "RandomFeatures", *args, **kwargs) -> None:
        if not module.training or not module.feature_redraw_interval:
            return 

        module.redraw_step.add_(1)
        if module.redraw_step >= module.feature_redraw_interval:
            module.redraw_projection_matrix()
            module.redraw_step.fill_(0)

class OrthogonalFeatures(RandomFeatures):

    @torch.no_grad()
    def create_projection(self, **kwargs) -> Tensor:
        r"""Creates a projection matrix using positive Gaussian orthogonal random features."""
        D, R = self.d, self.num_features // self.num_heads
        num_blocks = self.num_heads
        blocks: List[Tensor] = []
        for _ in range(num_blocks):
            mat = _sample_orth_matrix(R, self.var, **kwargs)
            blocks.append(mat)
        mat = torch.cat(blocks, 0) 
        mat = mat[:]

