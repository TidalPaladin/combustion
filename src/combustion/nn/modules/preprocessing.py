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


    Shape:
        - Inputs: Tensor of shape :math:`(*)` where :math:`*` indicates
          an arbitrary number of dimensions.
        - Output: Same shape as input.
    """

    def __init__(self, dims: Union[int, Tuple[int]], epsilon=1e-9):
        super(Standardize, self).__init__()
        if isinstance(dims, int):
            dims = (dims,)
        self.dims = set([int(x) for x in dims])
        self.epsilon = abs(float(epsilon))

    def __repr__(self):
        s = f"Standardize(dims={tuple(self.dims)}"
        if self.epsilon != 1e-9:
            s += f", epsilon={self.epsilon}"
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

        std, mean = torch.std_mean(inputs, dim=tuple(self.dims), keepdim=True)
        var = std.pow(2).clamp_(min=self.epsilon)
        result = inputs.sub(mean).div_(var)
        return result
