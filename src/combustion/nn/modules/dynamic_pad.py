#!/usr/bin/env python
# -*- coding: utf-8 -*-

from math import ceil, floor
from typing import Dict, Iterable, List, Optional, Tuple, Union

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


class MatchShapes(nn.Module):
    r"""Helper module that assists in checking and matching the spatial dimensions of two tensors.

    .. note::
        This function cannot fix mismatches along the batch/channel dimensions, nor can it fix tensors
        with an unequal number of dimensions.

    Args:

        strategy (str):
            Approach to matching unequal dimensions. Should be one of ``pad``, ``crop``.

        padding_mode (str):
            See :func:`torch.nn.functional.pad`

        fill_value (float):
            Fill value when using constant padding. See :func:`torch.nn.functional.pad`

        check_only (bool):
            If true, mismatched spatial dimensions will not be fixed and instead raise an exception.

    Shapes:
        * ``first`` - :math:`(B, C, *)`
        * ``second`` - :math:`(B, C, *)`
    """

    def __init__(
        self, strategy: str = "pad", padding_mode: str = "constant", fill_value: float = 0.0, check_only: bool = False,
    ):
        super().__init__()
        strategy = str(strategy).lower()
        if str(strategy) not in ("pad", "crop"):
            raise ValueError(f"Expected strategy in ['crop', 'pad'] but found {strategy}")
        padding_mode = str(padding_mode).lower()
        if padding_mode not in ["constant", "reflect", "replicate", "circular"]:
            raise ValueError(f"Unexpected padding mode `{padding_mode}`")
        if not isinstance(check_only, bool):
            raise TypeError(f"Expected bool for `check_only` but found {type(check_only)}")

        self._strategy = strategy
        self._padding_mode = padding_mode
        self._fill_value = float(fill_value)
        self._check_only = bool(check_only)

    def extra_repr(self):
        s = f"strategy={self._strategy}"
        if self._strategy == "pad":
            s += ", padding_mode={self._padding_mode}"
            if self._padding_mode == "constant":
                s += ", fill_value={self._fill_value}"
        if self._check_only:
            s += f"check_only={self._check_only}"
        return s

    def forward(self, first: Tensor, second: Tensor) -> Tuple[Tensor, Tensor]:
        if first.ndim != second.ndim:
            raise RuntimeError(f"Expected first.ndim == second.ndim, but found {first.ndim}, {second.ndim}")

        if first.shape[:2] != second.shape[:2]:
            raise RuntimeError(f"Shape mismatch along batch/channel dimension: {first.shape} vs {second.shape}")

        first_shape = first.shape[2:]
        second_shape = second.shape[2:]
        if first_shape == second_shape:
            return first, second

        if self._check_only:
            raise RuntimeError(f"Shape mismatch: {first.shape} vs {second.shape}")

        if self._strategy == "pad":
            return self._pad(first, second)
        elif self._strategy == "crop":
            return self._crop(first, second)
        else:
            raise NotImplementedError("Strategy {self._strategy}")

    def _crop(self, first: Tensor, second: Tensor) -> Tuple[Tensor, Tensor]:
        assert first.ndim == second.ndim
        first_shape = first.shape[2:]
        second_shape = second.shape[2:]

        for dim, (shape1, shape2) in enumerate(zip(first_shape, second_shape)):
            # skip over batch/channel dim
            dim = 2 + dim

            # get low/high crop indices
            start = abs(shape1 - shape2) // 2
            length = min(shape1, shape2)

            # crop
            if shape1 < shape2:
                second = second.narrow(dim, start, length)
            elif shape2 < shape1:
                first = first.narrow(dim, start, length)

        assert first.shape[2:] == second.shape[2:]
        return first, second

    def _pad(self, first: Tensor, second: Tensor) -> Tuple[Tensor, Tensor]:
        assert first.ndim == second.ndim
        first_shape = first.shape[2:]
        second_shape = second.shape[2:]

        first_padding: List[int] = [0,] * (2 * len(first_shape))
        second_padding: List[int] = [0,] * (2 * len(second_shape))

        for i, (shape1, shape2) in enumerate(zip(first_shape, second_shape)):
            low = floor(float(abs(shape1 - shape2)) / 2)
            high = ceil(float(abs(shape1 - shape2)) / 2)

            if shape1 < shape2:
                first_padding[2 * i] = low
                first_padding[2 * i + 1] = high
            elif shape2 < shape1:
                second_padding[2 * i] = low
                second_padding[2 * i + 1] = high

        # if sum(first_padding):
        first = F.pad(first, first_padding, self._padding_mode, self._fill_value)
        # if sum(second_padding):
        second = F.pad(second, second_padding, self._padding_mode, self._fill_value)

        assert first.shape[2:] == second.shape[2:]
        return first, second


PATCH_TYPES = [
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.MaxPool1d,
    nn.MaxPool2d,
    nn.MaxPool3d,
    nn.AvgPool1d,
    nn.AvgPool2d,
    nn.AvgPool3d,
]


def patch_dynamic_same_pad(
    module: nn.Module,
    padding_mode: str = "constant",
    pad_value: float = 0.0,
    include_classes: Iterable[type] = [],
    include_names: Iterable[str] = [],
    exclude_names: Iterable[str] = [],
) -> Dict[str, nn.Module]:
    r"""Patches spatial layers in a :class:`torch.nn.Module`, wrapping each layer in a
    :class:`combustion.nn.DynamicSamePad` module. This method allows for dynamic same padding
    to be added to a module during or after instantiation.

    .. note::
        This method alone is not sufficient to ensure shape matching throughout a U-Net or similar
        architecture. Use this method in conjunction with :class:`combustion.nn.MatchShapes` for
        correct end to end operation of any input.

    .. warning::
        This method is experimental

    Args:

        module (:class:`torch.nn.Module`):
            The module to patch with dynamic same padding.

        padding_mode (str):
            Padding mode for :class:`combustion.nn.DynamicSamePad`

        pad_value (str):
            Fill value for :class:`combustion.nn.DynamicSamePad`

        include_classes (iterable of types):
            Types of modules to be patched. By default, PyTorch's convolutional and
            pooling layers are matched

        include_names (iterable of str):
            Explicit names of children to be patched. If ``include_names`` is specified,
            only children whose names appear in ``include_names`` will be patched.

        exclude_names (iterable of str):
            Names of children to be excluded from patching.

    Returns:
        A mapping of child module names to their newly patched module instances.
    """
    if not include_classes:
        include_classes = PATCH_TYPES
    kwargs = {
        "padding_mode": padding_mode,
        "pad_value": pad_value,
        "include_classes": include_classes,
        "exclude_names": exclude_names,
    }

    patched: Dict[str, nn.Module] = {}

    # checks if a module is a direct patching target
    def is_patchable(module, module_name):
        if type(module) not in include_classes:
            return False
        if include_names:
            return module_name in include_names
        else:
            return module_name not in exclude_names

    for child_name, child in module.named_children():
        # patch this child if matches a target name/class
        if is_patchable(child, child_name):
            padded = DynamicSamePad(child, padding_mode, pad_value)
            setattr(module, child_name, padded)
            patched[child_name] = padded

        # recurse on patchable subchildren
        patched_children = patch_dynamic_same_pad(child, **kwargs)
        for k, v in patched_children.items():
            patched[f"{child_name}.{k}"] = v

    return patched
