#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from torch import Tensor
from typing import Callable, Optional
from torch.distributions.normal import Normal

KernelFunction = Callable[[Tensor], Tensor]

def generalized_kernel_features(
    data: Tensor,
    kernel_func: KernelFunction,
    projection: Optional[Tensor] = None,
    normalize: bool = True,
    kernel_epsilon: float = 1e-3
):
    r"""Constructs kernel features for fast generalized attention.

    Shapes:
        * ``data`` - :math:`(L, N, E)`
        * ``projection`` - :math:`(E, R)`
        Output - :math:`(L, N, R)` if ``projection`` is not ``None``, otherwise :math:`(L, N, E)`
    """

    L, N, E = data.shape

    if normalize:
        normalizer = data.new_tensor(E).pow(1/4).reciprocal()
        data = data * normalizer

    if projection is not None:
        R = projection.shape[-1]
        projection = projection.view(1, E, R).expand(N, -1, -1)
        assert projection.shape == (N, E, R)
        assert data.shape == (L, N, E)

        data = torch.einsum("lne,ner->lnr", data, projection)
        assert data.shape == (L, N, R)

    data = kernel_func(data) + kernel_epsilon
    return data

def nonnegative_softmax_features(
    data: Tensor,
    normalize: bool = True,
    kernel_epsilon: float = 1e-3
):
    r"""Constructs kernel features for fast generalized attention.

    Shapes:
        * ``data`` - :math:`(L, N, E)`
        * ``projection`` - :math:`(E, R)`

    """
    L, N, E = data.shape
    data = generalized_kernel_features()

    if normalize:
        normalizer = data.new_tensor(E).pow(1/4).reciprocal()
        data = data * normalizer

    if projection is not None:
        R = projection.shape[-1]
        projection = projection.view(1, E, R).expand(N, -1, -1)
        assert projection.shape == (N, E, R)
        assert data.shape == (L, N, E)

        data = torch.einsum("lne,ner->lnr", data, projection)
        assert data.shape == (L, N, R)

    data = kernel_func(data) + kernel_epsilon
    return data


class GaussianOrthogonalRandomMatrix(nn.Module):
  r"""Class providing a method to create Gaussian orthogonal matrix.
  Class is responsible for constructing 2D Gaussian orthogonal arrays.
  """

  def __init__(self, rows: int, cols: int, key, scaling=0):
    self.rows = rows
    self.cols = cols
    self.key = key
    self.scaling = scaling
    self.dist = Normal(0, 1)


    def forward(self) -> Tensor:
        H, W = self.rows, self.columns
        blocks = H // W

        block_list = []
        for _ in range(blocks):
            unstructured_block: Tensor = self.dist.rsample(torch.Size((H, W)))
            q, _ = unstructured_block.qr()
            q = q.T
            block_list.append(q)

        remaining_rows = H - blocks * W
        if remaining_rows > 0:
            unstructured_block: Tensor = self.dist.rsample(torch.Size((W, W)))
            q, _ = unstructured_block.qr()
            q = q.T
            block_list.append(q[:remaining_rows])

        final_matrix = torch.stack(block_list)

        if self.scaling == 0:
            multiplier = jnp.linalg.norm(
            random.normal(self.key, (self.nb_rows, self.nb_columns)), axis=1)
        elif self.scaling == 1:
            multiplier = jnp.sqrt(float(self.nb_columns)) * jnp.ones((self.nb_rows))
        else:
            raise ValueError('Scaling must be one of {0, 1}. Was %s' % self._scaling)

        return jnp.matmul(jnp.diag(multiplier), final_matrix)
