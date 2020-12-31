#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from .typing import Array


@torch.jit.ignore
def check_shapes_match(x: Array, y: Array, x_name: str, y_name: str) -> None:
    r"""Raises a :class:`ValueError` if two tensors do not have the same shape

    Args:
        x (numpy array or :class:`torch.Tensor`):
            One of the two tensors to check

        y (numpy array or :class:`torch.Tensor`):
            One of the two tensors to check

        x_name (str):
            Variable name for tensor ``x`` in the :class:`ValueError` text

        y_name (str):
            Variable name for tensor ``y`` in the :class:`ValueError` text
    """
    assert isinstance(x, (Tensor, np.ndarray))
    assert isinstance(y, (Tensor, np.ndarray))
    assert isinstance(x_name, str)
    assert isinstance(y_name, str)
    if x.shape != y.shape:
        raise ValueError(f"expected {x_name}.shape == {y_name}.shape:\n" f"{x.shape} vs {y.shape}")


@torch.jit.ignore
def check_ndim_match(x: Array, y: Array, x_name: str, y_name: str) -> None:
    r"""Raises a :class:`ValueError` if two tensors do not have the same number of dimensions

    Args:
        x (numpy array or :class:`torch.Tensor`):
            One of the two tensors to check

        y (numpy array or :class:`torch.Tensor`):
            One of the two tensors to check

        x_name (str):
            Variable name for tensor ``x`` in the :class:`ValueError` text

        y_name (str):
            Variable name for tensor ``y`` in the :class:`ValueError` text
    """
    assert isinstance(x, (Tensor, np.ndarray))
    assert isinstance(y, (Tensor, np.ndarray))
    assert isinstance(x_name, str)
    assert isinstance(y_name, str)
    if x.ndim != y.ndim:
        raise ValueError(f"expected {x_name}.ndim == {y_name}.ndim:\n" f"{x.shape} vs {y.shape}")


def check_is_tensor(x: Any, name: str) -> None:
    r"""Raises a :class:`TypeError` if the input is not a :class:`torch.Tensor`

    Args:
        x (Any):
            Input to check

        name (str):
            Variable name for input ``x`` in the :class:`TypeError` text
    """
    assert isinstance(name, str)
    if not isinstance(x, Tensor):
        raise TypeError(f"{name} must be type Tensor,\n" f"got {type(x)}")


@torch.jit.ignore
def check_is_array(x: Array, name: str) -> None:
    r"""Raises a :class:`TypeError` if the input is not a :class:`torch.Tensor` or
    :class:`numpy.ndarray`

    Args:
        x (Any):
            Input to check

        name (str):
            Variable name for input ``x`` in the :class:`TypeError` text
    """
    assert isinstance(name, str)
    if not isinstance(x, (Tensor, np.ndarray)):
        raise TypeError(f"{name} must be type Tensor or np.ndarray,\n" f"got {type(x)}")


@torch.jit.ignore
def check_shape(x: Array, shape: Union[Tuple[int, ...], List[int]], name: str) -> None:
    r"""Raises a :class:`ValueError` if the input does not have the expected shape.

    Args:
        x (numpy array or :class:`torch.Tensor`):
            Input to check

        shape (iterable of ``Optional[int]``):
            Expected shape of the tensor, or ``None`` for a dimension that should not be checked

        name (str):
            Variable name for input ``x`` in the :class:`ValueError` text
    """
    assert isinstance(x, (Tensor, np.ndarray))
    assert isinstance(shape, (list, tuple, torch.Size))
    assert isinstance(name, str)

    # check tensor rank mismatch first
    if x.ndim != len(shape):
        raise ValueError(f"expected {name}.shape == {shape}:\n" f"{x.shape} vs {shape}")

    # check each dim, skip None entries in shape
    for i, (actual, expected) in enumerate(zip(x.shape, shape)):
        if expected is None:
            continue
        elif expected != actual:
            raise ValueError(f"expected {name}.shape == {shape}:\n" f"mismatch in dim {i}: {x.shape} vs {shape}")


@torch.jit.ignore
def check_dimension(x: Array, dim: int, size: int, name: str) -> None:
    r"""Raises a :class:`ValueError` if a given dimension does not have the expected size.

    Args:
        x (numpy array or :class:`torch.Tensor`):
            Input to check

        dim (int):
            Dimension to check

        size (int):
            Expected dimension size

        name (str):
            Variable name for input ``x`` in the :class:`ValueError` text
    """
    assert isinstance(x, (Tensor, np.ndarray))
    dim = int(dim)
    size = int(size)
    assert isinstance(name, str)

    if x.shape[dim] != size:
        raise ValueError(f"expected {name}.shape[{dim}] == {size}\n" f"found shape {x.shape}")


@torch.jit.ignore
def check_ndim(x: Array, ndim: int, name: str) -> None:
    r"""Raises a :class:`ValueError` if a given dimension does not have the expected size.

    Args:
        x (numpy array or :class:`torch.Tensor`):
            Input to check

        ndim (int):
            Expected number of dimensions

        name (str):
            Variable name for input ``x`` in the :class:`ValueError` text
    """
    assert isinstance(x, (Tensor, np.ndarray))
    ndim = int(ndim)
    assert isinstance(name, str)

    if x.ndim != ndim:
        raise ValueError(f"expected {name}.ndim == {ndim}\n" f"found shape {x.ndim}")


@torch.jit.ignore
def check_dimension_within_range(x: Array, dim: int, bounds: Tuple[Optional[int], Optional[int]], name: str) -> None:
    r"""Raises a :class:`ValueError` if the size of a given dimension does not fall within
    an expected range.

    Args:
        x (numpy array or :class:`torch.Tensor`):
            Input to check

        dim (int):
            Dimension to check

        bounds (tuple of two ints):
            Lower and upper bound of expected dimension sizes. Bounds can be ``None`` for
            no constraint

        name (str):
            Variable name for input ``x`` in the :class:`ValueError` text
    """
    assert isinstance(x, (Tensor, np.ndarray))
    dim = int(dim)
    assert isinstance(name, str)

    low = bounds[0] if bounds[0] is not None else 0
    high = bounds[1] if bounds[1] is not None else float("inf")

    if x.shape[dim] < low or x.shape[dim] > high:
        raise ValueError(f"expected {low} < {name}.shape[{dim}] < {high}\n" f"found shape {x.shape}")


@torch.jit.ignore
def check_ndim_within_range(x: Array, bounds: Tuple[Optional[int], Optional[int]], name: str) -> None:
    r"""Raises a :class:`ValueError` if a given dimension does not have the expected size.

    Args:
        x (numpy array or :class:`torch.Tensor`):
            Input to check

        bounds (tuple of two ints):
            Lower and upper bound of expected number of dimensions. Bounds can be ``None`` for
            no constraint

        name (str):
            Variable name for input ``x`` in the :class:`ValueError` text
    """
    assert isinstance(x, (Tensor, np.ndarray))
    assert isinstance(name, str)

    low = bounds[0] if bounds[0] is not None else 0
    high = bounds[1] if bounds[1] is not None else float("inf")

    if x.ndim < low or x.ndim > high:
        raise ValueError(f"expected {low} < {name}.ndim < {high}\n" f"found shape {x.ndim}")


@torch.jit.ignore
def check_dimension_match(x: Array, y: Array, dim: int, x_name: str, y_name: str) -> None:
    r"""Raises a :class:`ValueError` if two tensors have different sizes for a given dimension.

    Args:
        x (numpy array or :class:`torch.Tensor`):
            One of the two tensors to check

        y (numpy array or :class:`torch.Tensor`):
            One of the two tensors to check

        dim (int):
            Dimension to check

        x_name (str):
            Variable name for tensor ``x`` in the :class:`ValueError` text

        y_name (str):
            Variable name for tensor ``y`` in the :class:`ValueError` text
    """
    assert isinstance(x, (Tensor, np.ndarray))
    assert isinstance(y, (Tensor, np.ndarray))
    dim = int(dim)
    assert isinstance(x_name, str)
    assert isinstance(y_name, str)

    if x.shape[dim] != y.shape[dim]:
        raise ValueError(f"expected {x_name}.shape[{dim}] == {y_name}.shape[{dim}]\n" f"{x.shape} vs {y.shape}")


@torch.jit.ignore
def check_names(x: Array, names: Union[List[str], Tuple[str, ...]], var_name) -> None:
    assert isinstance(x, (Tensor, np.ndarray))
    assert isinstance(names, (list, tuple))
    assert isinstance(var_name, str)

    expected = sorted(names)
    actual = sorted(x.names)
    if not actual == expected:
        raise ValueError(f"expected {var_name}.names == {names}:\n" f"{actual} vs {expected}")


@torch.jit.ignore
def check_names_match(x: Array, y: Array, x_name: str, y_name: str):
    assert isinstance(x, (Tensor, np.ndarray))
    assert isinstance(y, (Tensor, np.ndarray))
    assert isinstance(x_name, str)
    assert isinstance(y_name, str)

    x_names = sorted(x.names)
    y_names = sorted(y.names)
    if not x_names == y_names:
        raise ValueError(f"expected {x_name}.names == {y_name}.names:\n" f"{x_names} vs {y_names}")
