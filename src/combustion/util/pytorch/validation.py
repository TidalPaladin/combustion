#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List, Tuple, Union

import numpy as np
from torch import Tensor

from .typing import Array


def check_shapes_match(x: Array, y: Array, x_name: str, y_name: str):
    assert isinstance(x, (Tensor, np.ndarray))
    assert isinstance(y, (Tensor, np.ndarray))
    assert isinstance(x_name, str)
    assert isinstance(y_name, str)
    if x.shape != y.shape:
        raise ValueError(f"expected {x_name}.shape == {y_name}.shape:\n" "{x.shape} vs {y.shape}")


def check_ndim_match(x: Array, y: Array, x_name: str, y_name: str):
    assert isinstance(x, (Tensor, np.ndarray))
    assert isinstance(y, (Tensor, np.ndarray))
    assert isinstance(x_name, str)
    assert isinstance(y_name, str)
    if x.ndim != y.ndim:
        raise ValueError(f"expected {x_name}.ndim == {y_name}.ndim:\n" "{x.shape} vs {y.shape}")


def check_names_match(x: Array, y: Array, x_name: str, y_name: str):
    assert isinstance(x, (Tensor, np.ndarray))
    assert isinstance(y, (Tensor, np.ndarray))
    assert isinstance(x_name, str)
    assert isinstance(y_name, str)

    x_names = sorted(x.names)
    y_names = sorted(y.names)
    if not x_names == y_names:
        raise ValueError(f"expected {x_name}.names == {y_name}.names:\n" "{x_names} vs {y_names}")


def check_is_tensor(x: Array, name: str):
    assert isinstance(name, str)
    if not isinstance(x, (Tensor, np.ndarray)):
        raise TypeError(f"{name} must be type Tensor or np.ndarray,\n" "got {type(x)}")


def check_shape(x: Array, shape: Union[Tuple[int, ...], List[int]], name: str):
    assert isinstance(x, (Tensor, np.ndarray))
    assert isinstance(shape, [list, tuple])
    assert isinstance(name, str)

    # check tensor rank mismatch first
    if x.ndim != len(shape):
        raise ValueError(f"expected {name}.shape == {shape}:\n" "{x.shape} vs {shape}")

    # check each dim, skip None entries in shape
    for i, (actual, expected) in enumerate(zip(x.shape, shape)):
        if expected is None:
            continue
        raise ValueError(f"expected {name}.shape == {shape}:\n" "mismatch in dim {i}: {x.shape} vs {shape}")


def check_names(x: Array, names: Union[List[str], Tuple[str, ...]], var_name):
    assert isinstance(x, (Tensor, np.ndarray))
    assert isinstance(names, [list, tuple])
    assert isinstance(var_name, str)

    expected = sorted(names)
    actual = sorted(x.names)
    if not actual == expected:
        raise ValueError(f"expected {var_name}.names == {names}:\n" "{actual} vs {expected}")
