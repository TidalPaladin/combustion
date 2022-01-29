#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
from math import ceil, floor
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from combustion.exceptions import ShapeMismatchError

from ..functional.fill_masked import fill_normal


class MatchShapes(nn.Module):
    r"""Helper module that assists in checking and matching the spatial dimensions of tensors.

    When given a list of tensors, matches each spatial dimension according to the minimum or maximum size
    among tensors, depending on whether padding or cropping is requested. When given an expicit shape, spatial
    dimensions are padded / cropped to match the target shape.

    Raises :class:`combustion.exceptions.ShapeMismatchError` when shapes cannot be matched.

    .. note::
        This function cannot fix mismatches along the batch/channel dimensions, nor can it fix tensors
        with an unequal number of dimensions.

    .. warning::
        This module is compatible with TorchScript scripting, but may have incorrect behavior when traced.

    Args:

        strategy (str):
            Approach to matching unequal dimensions. Should be one of ``pad``, ``crop``.

        ignore_channels (bool):
            If true, allow mismatch in channel dimension (e.g. for concatenation).

        padding_mode (str):
            * ``"constant"`` - Pad with a constant value
            * ``"replicate"`` - Pad by replicating image edges
            * ``"reflect"`` - Pad by reflecting image about edges
            * ``"mean"`` - Pad with the per-channel mean
            * ``"var_mean"`` - Pad with noise sampled from normal distribution with per-channel
              mean and variance

        fill_value (float):
            Fill value when using constant padding. See :func:`torch.nn.functional.pad`

        check_only (bool):
            If true, mismatched spatial dimensions will not be fixed and instead raise an exception.

        warn_pct_change (float):
            Value in the range :math:`[0, 1]`. If a spatial dimension is resized such that the ratio of
            old to new sizes exceeds ``warn_pct_change``, a warning will be raised.

    Shapes:
        * ``tensors`` - :math:`(B, C_i, *)`

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
        self,
        strategy: str = "pad",
        ignore_channels: bool = False,
        padding_mode: str = "constant",
        fill_value: float = 0.0,
        check_only: bool = False,
        warn_pct_change: float = 100.0,
    ):
        super().__init__()
        strategy = str(strategy).lower()
        if str(strategy) not in ("pad", "crop"):
            raise ValueError(f"Expected strategy in ['crop', 'pad'] but found {strategy}")
        padding_mode = str(padding_mode).lower()
        if padding_mode not in ["constant", "reflect", "replicate", "circular", "mean", "var_mean"]:
            raise ValueError(f"Unexpected padding mode `{padding_mode}`")
        if not isinstance(check_only, bool):
            raise TypeError(f"Expected bool for `check_only` but found {type(check_only)}")

        self._strategy = strategy
        self._padding_mode = padding_mode
        self._fill_value = float(fill_value)
        self._check_only = bool(check_only)
        self._ignore_channels = bool(ignore_channels)
        self._warn_pct_change = float(warn_pct_change)

    def extra_repr(self):
        s = f"strategy='{self._strategy}'"
        if self._ignore_channels:
            s += "ignore_channels=True"
        if self._strategy == "pad":
            s += f", padding_mode='{self._padding_mode}'"
            if self._padding_mode == "constant":
                s += f", fill_value={self._fill_value}"
        if self._check_only:
            s += "check_only=True"
        if self._warn_pct_change != 100.0:
            s += f"warn_pct_change={self._warn_pct_change}"
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
                if not isinstance(val, (float, int, Tensor)):
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
            if self._check_only and not self._has_shape(tensor, target_shape):
                raise ShapeMismatchError(
                    f"Shape mismatch at position {i}: expected {target_shape}, found {tensor.shape}"
                )
            if first_tensor.shape[0] != tensor.shape[0]:
                raise ShapeMismatchError(
                    f"Expected batch dimensions == {target_shape[0]} for all tensors, "
                    f"but found B = {tensor.shape[0]} at position {i}"
                )
            if not self._ignore_channels and first_tensor.shape[1] != tensor.shape[1]:
                raise ShapeMismatchError(
                    f"Expected channel dimensions == {target_shape[1]} for all tensors, "
                    f"but found C = {tensor.shape[1]} at position {i}"
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

        tensor_padding: List[int] = [
            0,
        ] * (2 * len(tensor_shape))
        second_padding: List[int] = [
            0,
        ] * (2 * len(spatial_shape))
        has_padding = False

        self._warn_on_extreme_change(tensor, shape)
        for i, raw_shape in enumerate(tensor_shape):
            padded_shape: int = spatial_shape[i]
            if raw_shape >= padded_shape:
                continue

            low = floor(float(abs(padded_shape - raw_shape)) / 2)
            high = ceil(float(abs(padded_shape - raw_shape)) / 2)
            tensor_padding[-(2 * i + 1)] = low
            tensor_padding[-(2 * i + 2)] = high
            has_padding = True

        if not has_padding:
            return tensor

        # for mean/var_mean, per channel padding is needed.
        # create a mask and fill the padded tensor at mask positions
        if self._padding_mode in ["mean", "var_mean"]:
            padded_mask = F.pad(torch.zeros_like(tensor), tensor_padding, mode="constant", value=1.0).to(
                dtype=torch.bool
            )
            tensor = F.pad(tensor, tensor_padding, mode="constant", value=0.0)
            preserve_var = self._padding_mode == "var_mean"
            tensor = fill_normal(tensor, fill_mask=padded_mask, sample_mask=~padded_mask, preserve_var=preserve_var)
        # otherwise pad normally
        else:
            tensor = F.pad(tensor, tensor_padding, self._padding_mode, self._fill_value)

        return tensor

    def _warn_on_extreme_change(self, tensor: Tensor, shape: List[int]) -> None:
        for src, target in zip(tensor.shape[2:], shape[2:]):
            ratio = max(src // target, target // src)
            if 1.0 / ratio > self._warn_pct_change:
                warnings.warn(
                    f"Resized a tensor dimension by >= {self._warn_pct_change * 100}% "
                    f"matching {tensor.shape} to tuple({shape})"
                )
                return

    def _has_shape(self, tensor: Tensor, shape: List[int]) -> bool:
        if tensor.ndim != len(shape):
            return False

        for i, size in enumerate(shape):
            if i == 1 and self._ignore_channels:
                continue
            if tensor.shape[i] != size:
                return False
        return True
