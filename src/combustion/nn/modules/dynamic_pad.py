#!/usr/bin/env python
# -*- coding: utf-8 -*-

from math import ceil, floor
from typing import List, Tuple, Union

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from combustion.util import double, single, triple


class DynamicSamePad(nn.Module):
    r"""Wraps a :class:`torch.nn.Module` with ``kernel_size`` and ``stride`` attributes, dynamically padding
    the input similar to TensorFlow's "same" padding. For non-unit strides, padding is applied such that the
    padded input size is a multiple of the stride.

    Args:
        module (:class:`torch.nn.Module`):
            The module to wrap. Should have ``kernel_size`` and ``stride`` attributes, and accept a single
            input tensor. Tested with PyTorch's convolutional / padding layers.

        padding_mode (str):
            ``'constant'``, ``'reflect'``, ``'replicate'``, or ``'circular'``. Default ``'constant'``

        pad_value (str):
            Fill value for ``'constant'`` padding.  Default ``0``
    """

    def __init__(self, module: nn.Module, padding_mode: str = "constant", pad_value: float = 0.0):
        super().__init__()
        if not hasattr(module, "kernel_size"):
            raise AttributeError(f"Expected {module.__class__.__name__} to have `kernel_size` attribute")
        if not hasattr(module, "stride"):
            raise AttributeError(f"Expected {module.__class__.__name__} to have `stride` attribute")

        self._module = module
        self._stride = self._to_tuple(module, module.stride)
        self._kernel_size = self._to_tuple(module, module.kernel_size)

        padding_mode = str(padding_mode).lower()
        if padding_mode not in ["constant", "reflect", "replicate", "circular"]:
            raise ValueError(f"Unexpected padding mode `{padding_mode}`")
        self._padding_mode = padding_mode
        self._padding_value = float(pad_value)

        # override module's padding if set
        module.padding = (0,) * len(self._to_tuple(module, module.padding))

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim < 3:
            raise ValueError(f"Expected inputs.ndim >= 3, but found {inputs.ndim}")

        unpadded_dim_shapes = inputs.shape[2:]
        inputs.ndim - 2
        stride = self._stride
        kernel_size = self._kernel_size

        # get padding amount on both edges for each dim in input
        padding: List[int] = []
        for i, (s, k) in enumerate(zip(stride, kernel_size)):
            dim_shape = int(unpadded_dim_shapes[i])
            # pad to maintain size based on kernel_size + ensure padded is multiple of stride
            low = k // 2 + floor(dim_shape % s / s)
            high = k // 2 + ceil(dim_shape % s / s)
            padding.append(low)
            padding.append(high)

        # pad and pass padded input to wrapped module
        padded_input = F.pad(inputs, padding, self._padding_mode, self._padding_value)
        return self._module(padded_input)

    def _to_tuple(self, module: nn.Module, val: Union[Tuple[int], int]) -> Tuple[int]:
        if isinstance(val, tuple):
            return val

        module_name = module.__class__.__name__.lower()
        if "1d" in module_name:
            return single(val)
        elif "2d" in module_name:
            return double(val)
        elif "3d" in module_name:
            return triple(val)
        else:
            raise ValueError(f"Couldn't infer tuple size for class {module.__class__.__name__}")
