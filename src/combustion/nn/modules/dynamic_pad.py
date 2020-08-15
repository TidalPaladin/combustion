#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
from math import ceil, floor
from typing import Dict, Iterable, List, Optional, Tuple, Union

import torch
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


class ShapeMismatchError(RuntimeError):
    r"""Raised on failure of :class:`combustion.nn.MatchShapes`."""


class MatchShapes(nn.Module):
    r"""Helper module that assists in checking and matching the spatial dimensions of tensors.

    When given a list of tensors, matches each spatial dimension according to the minimum or maximum size
    among tensors, depending on whether padding or cropping is requested. When given an expicit shape, spatial
    dimensions are padded / cropped to match the target shape.

    Raises :class:`combustion.exceptions.ShapeMismatchError` when shapes cannot be matched.

    .. note::
        This function cannot fix mismatches along  the batch/channel dimensions, nor can it fix tensors
        with an unequal number of dimensions.

    .. warning::
        This module is compatible with TorchScript scripting, but may have incorrect behavior when traced.

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
        * ``tensors`` - :math:`(B, C, *)`

    Basic Example::

        >>> t1 = torch.rand(1, 1, 10, 10)
        >>> t2 = torch.rand(1, 1, 13, 13)
        >>> layer = MatchShapes()
        >>> pad1, pad2 = layer([t1, t2])

    Explicit Shape Example::

        >>> t1 = torch.rand(1, 1, 10, 10)
        >>> t2 = torch.rand(1, 1, 13, 13)
        >>> layer = MatchShapes()
        >>> pad1, pad2 = layer([t1, t2], shape_override=(12, 12))
    """

    def __init__(
        self, strategy: str = "pad", padding_mode: str = "constant", fill_value: float = 0.0, check_only: bool = False
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

    def forward(self, tensors: List[Tensor], shape_override: Optional[List[int]] = None) -> List[Tensor]:
        r"""Matches the shapes of all tensors in a list, with an optional explicit shape override

        Args:
            tensors (list of :class:`torch.Tensor`):
                The tensors to match shapes of

            shape_override (iterable of ints, optional):
                By default the target shape is chosen based on tensor sizes and the strategy (cropping/padding).
                Setting ``shape_override`` sets an explicit output shape, and padding/cropping is chosen on a
                per-dimension basis to satisfy this target shape. Overrides ``strategy``. Should only include
                spatial dimensions (not batch/channel sizes).

        """
        if isinstance(tensors, Tensor):
            tensors = [
                tensors,
            ]
        # check tensors has at least one element and extract the first tensor
        if not len(tensors):
            raise ValueError("`tensors` must be a non-empty list of tensors")
        first_tensor = tensors[0]

        # use the explicit shape override if given, or use first tensor's shape as an initial target
        if shape_override is not None:
            new_shape_override = list(shape_override)
            for i, val in enumerate(shape_override):
                if not isinstance(val, (float, int)):
                    raise TypeError(f"Expected float or int for shape_override at pos {i} but found {type(val)}")
                new_shape_override[i] = int(val)
            new_shape_override = list(first_tensor.shape[:2]) + new_shape_override
            target_shape = list(new_shape_override)
        elif len(tensors) < 2:
            raise ValueError("`shape_override` must be specified when `tensors` contains only one tensor")
        else:
            target_shape = list(first_tensor.shape)

        # validate required matches in ndim / batch / channel
        for i, tensor in enumerate(tensors[1:]):
            if tensor.ndim != len(target_shape):
                raise ShapeMismatchError(
                    f"Expected tensor.ndim == {len(target_shape)} for all tensors, "
                    f"but found {tensor.ndim} at position {i}"
                )
            if self._check_only and target_shape != tensor.shape:
                raise ShapeMismatchError(
                    f"Shape mismatch at position {i}: expected {target_shape}, found {tensor.shape}"
                )
            if first_tensor.shape[:2] != tensor.shape[:2]:
                raise ShapeMismatchError(
                    f"Expected batch, channel dimensions == {target_shape[:2]} for all tensors, "
                    f"but found (B, C) = {tensor.shape[2:]} at position {i}"
                )

        # if explicit shape wasn't given, need to update spatial dims of target shape according to strategy
        if shape_override is None:
            for i in range(2, len(target_shape)):
                biggest_size = 0
                smallest_size = 2 ** 60
                for tensor in tensors:
                    biggest_size = max(tensor.shape[i], biggest_size)
                    smallest_size = min(tensor.shape[i], smallest_size)

                # when padding, pad to the largest size for dim i among all tensors
                if self._strategy == "pad":
                    target_shape[i] = int(biggest_size)
                # when cropping, crop to the smallest size for dim i among all tensors
                elif self._strategy == "crop":
                    target_shape[i] = int(smallest_size)
                else:
                    raise NotImplementedError("Strategy {self._strategy}")

        # pad/crop each tensor to the correct target shape
        #
        # since we might not satisfy shape_override using cropping/padding alone, try user specified strategy
        # first then fallback to alternate strategy
        for i, tensor in enumerate(tensors):
            if self._strategy == "pad":
                new_tensor = self._pad(tensor, target_shape)
                if new_tensor.shape != target_shape:
                    new_tensor = self._crop(new_tensor, target_shape)
                tensors[i] = new_tensor
            elif self._strategy == "crop":
                new_tensor = self._crop(tensor, target_shape)
                if new_tensor.shape != target_shape:
                    new_tensor = self._pad(new_tensor, target_shape)
                tensors[i] = new_tensor
            else:
                raise NotImplementedError("Strategy {self._strategy}")
            assert tensors[i].shape == torch.Size(target_shape)

        return tensors

    def _crop(self, tensor: Tensor, shape: List[int]) -> Tensor:
        assert tensor.ndim == len(shape)
        tensor_shape = tensor.shape[2:]
        spatial_shape = shape[2:]

        self._warn_on_extreme_change(tensor, shape)
        for dim, (raw_shape, cropped_shape) in enumerate(zip(tensor_shape, spatial_shape)):
            if raw_shape <= cropped_shape:
                continue

            # skip over batch/channel dim
            dim = 2 + dim

            # get low/high crop indices
            start = abs(raw_shape - cropped_shape) // 2
            length = min(raw_shape, cropped_shape)

            tensor = tensor.narrow(dim, start, length)

        return tensor

    def _pad(self, tensor: Tensor, shape: List[int]) -> Tensor:
        assert tensor.ndim == len(shape)
        tensor_shape = tensor.shape[2:]
        spatial_shape = shape[2:]

        tensor_padding: List[int] = [0,] * (2 * len(tensor_shape))
        second_padding: List[int] = [0,] * (2 * len(spatial_shape))
        has_padding = False

        self._warn_on_extreme_change(tensor, shape)
        for i, raw_shape in enumerate(tensor_shape):
            padded_shape: int = spatial_shape[i]
            if raw_shape >= padded_shape:
                continue

            low = floor(float(abs(padded_shape - raw_shape)) / 2)
            high = ceil(float(abs(padded_shape - raw_shape)) / 2)
            tensor_padding[2 * i] = low
            tensor_padding[2 * i + 1] = high
            has_padding = True

        if has_padding:
            tensor = F.pad(tensor, tensor_padding, self._padding_mode, self._fill_value)

        return tensor

    def _warn_on_extreme_change(self, tensor: Tensor, shape: List[int]) -> None:
        for src, target in zip(tensor.shape, shape):
            if src / target >= 2 or src / target <= 0.5:
                warnings.warn(f"Resized a tensor dimension by >= 50% matching {tensor.shape} to tuple({shape})")
                return


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
