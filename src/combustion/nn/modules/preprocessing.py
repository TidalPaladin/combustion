#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Tuple, Union

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

        # determine shape properties and which dims should be reduced / preserved
        ndim = inputs.ndim
        inputs.shape
        reduce_dims = set([d if d >= 0 else ndim + d for d in self.dims])
        preserve_dims = set(range(ndim)) - reduce_dims

        # flatten all reduced dimensions into dim=-1
        inputs = inputs.permute(*preserve_dims, *reduce_dims)
        permuted_shape = inputs.shape
        inputs = inputs.flatten(start_dim=-1 * len(reduce_dims))

        # compute mean /variance over reduced dimensions and standardize input
        mean = inputs.mean(dim=-1, keepdim=True)
        var = inputs.var(dim=-1, keepdim=True).clamp_(min=self.epsilon)
        result = inputs.sub_(mean).div_(var)

        # restore original shape
        return result.view(*permuted_shape).permute(*preserve_dims, *reduce_dims)
