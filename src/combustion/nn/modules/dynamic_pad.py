#!/usr/bin/env python
# -*- coding: utf-8 -*-

from math import ceil, floor
from typing import List, Optional, Tuple, Union

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from combustion.util import double, single, triple


class DynamicSamePad(nn.Module):
    r"""Wraps a :class:`torch.nn.Module`, dynamically padding the input similar to TensorFlow's "same" padding.
    For non-unit strides, padding is applied such that the padded input size is a multiple of the stride.

    By default, ``kernel_size`` and ``stride`` are determined by accessing the corresponding attributes on
    the :class:`torch.nn.Module`. When these attributes are not present, they can be passed explicitly
    to this module's constructor.

    This module is robust to modules of different dimensionalities (e.g. 1d, 2d, 3d). The dimensionality
    is determined using the following precedence:
        * If a ``kernel_size`` or ``stride`` override is passed with a tuple input, the length of the tuple
          determines the dimensionality.

        * If ``kernel_size`` and ``stride`` attributes on ``module`` are tuples, the length of these tuples
          determines the dimensionality.

        * The dimensionality is determined by comparing ``module.__class__.__name__.lower()`` against
          ``['1d', '2d', '3d']``.

        * No options remain, and ``ValueError`` is raised.

    .. warning::
        This module is compatible with TorchScript scripting, but may have incorrect behavior when traced.

    Args:
        module (:class:`torch.nn.Module`):
            The module to wrap. Should have ``kernel_size`` and ``stride`` attributes, and accept a single
            input tensor. Tested with PyTorch's convolutional / padding layers.

        padding_mode (str):
            ``'constant'``, ``'reflect'``, ``'replicate'``, or ``'circular'``. Default ``'constant'``

        pad_value (str):
            Fill value for ``'constant'`` padding.  Default ``0``

        kernel_size (int or tuple of ints):
            Explicit kernel size to use in padding calculation, overriding ``module.kernel_size`` if present.
            By default, ``kernel_size`` is set using ``module.kernel_size``.

        stride (int or tuple of ints):
            Explicit stride to use in padding calculation, overriding ``module.kernel_size`` if present.
            By default, ``stride`` is set using ``module.stride``.

    Shapes:
        * Input - :math:`(B, C, *)`

    Basic Example::

        >>> conv = torch.nn.Conv2d(1, 1, kernel_size=3, stride=2)
        >>> same_conv = DynamicSamePad(conv)
        >>> inputs = torch.rand(1, 1, 11, 11)
        >>> outputs = same_conv(inputs)
        >>> print(outputs.shape)

    Example Using Explicit Sizes::

        >>> conv = torch.nn.Conv2d(1, 1, kernel_size=3, stride=2)
        >>> # kernel_size / stride must be given if module doesn't have kernel_size/stride attributes
        >>> same_conv = DynamicSamePad(conv, kernel_size=3, stride=(2, 2))
        >>> inputs = torch.rand(1, 1, 11, 11)
        >>> outputs = same_conv(inputs)
        >>> print(outputs.shape)
    """

    def __init__(
        self,
        module: nn.Module,
        padding_mode: str = "constant",
        pad_value: float = 0.0,
        kernel_size: Optional[Union[Tuple[float], float]] = None,
        stride: Optional[Union[Tuple[float], float]] = None,
    ):
        super().__init__()
        name = module.__class__.__name__
        if not isinstance(module, nn.Module):
            raise TypeError(f"Expected module to be nn.Module, but found {name}")
        if kernel_size is None and not hasattr(module, "kernel_size"):
            raise AttributeError(f"Expected {name} to have `kernel_size` attribute or `kernel_size` param to be given")
        if not hasattr(module, "stride"):
            raise AttributeError(f"Expected {name} to have `stride` attribute or `stride` param to be given")

        self._module = module
        self._stride = self._to_tuple(module, stride if stride is not None else module.stride)
        self._kernel_size = self._to_tuple(module, kernel_size if kernel_size is not None else module.kernel_size)

        padding_mode = str(padding_mode).lower()
        if padding_mode not in ["constant", "reflect", "replicate", "circular"]:
            raise ValueError(f"Unexpected padding mode `{padding_mode}`")
        self._padding_mode = padding_mode
        self._padding_value = float(pad_value)

        # override module's padding if set
        module.padding = (0,) * len(self._to_tuple(module, module.padding))

    def extra_repr(self):
        s = f"padding_mode={self._padding_mode}"
        if self._padding_mode == "constant" and self._padding_value != 0:
            s += ", pad_value={self._padding_value}"
        return s

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
            raise ValueError(
                f"Couldn't infer tuple size for class {module.__class__.__name__}. " "Please pass an explicit tuple."
            )
