#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor


class Standardize(nn.Module):
    r"""Standardizes an input tensor to zero mean unit variance along
    one or more dimensions. Mean and variance will be computed over the
    selected dimensions, and the resultant tensor will be computed as

    .. math::
        x_o = \frac{x_i - \mu}{\max(\sigma^2, \epsilon)}

    Args:
        dims (int or tuple of ints): The dimension(s) to standardize over
        epsilon (float, optional): Lower bound on variance
        unbiased (bool, optional): Whether or not to used unbiased estimation
            in variance calculation. See :func:`torch.var_mean` for more details.

    Shape:
        - Inputs: Tensor of shape :math:`(*)` where :math:`*` indicates
          an arbitrary number of dimensions.
        - Output: Same shape as input.
    """

    def __init__(self, dims: Union[int, Tuple[int]], epsilon: float = 1e-9, unbiased: bool = True):
        super(Standardize, self).__init__()
        if isinstance(dims, int):
            dims = (dims,)
        self.dims = set([int(x) for x in dims])
        self.epsilon = abs(float(epsilon))
        self.unbiased = bool(unbiased)

    def __repr__(self):
        s = f"Standardize(dims={tuple(self.dims)}"
        if self.epsilon != 1e-9:
            s += f", epsilon={self.epsilon}"
        if not self.unbiased:
            s += f", unbiased={self.unbiased}"
        s += ")"
        return s

    def forward(self, inputs: Tensor) -> Tensor:
        r"""
        Args:
            inputs (Tensor): The tensor to be standardized
        """
        for dim in self.dims:
            if abs(dim) >= inputs.ndim:
                raise ValueError(f"Invalid dim {dim} for input of shape {inputs.shape}")

        var, mean = torch.var_mean(inputs, dim=tuple(self.dims), keepdim=True, unbiased=self.unbiased)
        var.clamp_(min=self.epsilon)
        result = inputs.sub(mean).div_(var)
        return result
