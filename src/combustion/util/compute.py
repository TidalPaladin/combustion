#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Union, Optional

from torch import Tensor

def slice_along_dim(x: Tensor, dim: int, start: Optional[int]=None, end: Optional[int] = None, step: Optional[int] = None):
    if dim < 0:
        dim = dim + x.ndim
    slices = [slice(0, None) if i != dim else slice(start, end, step) for i in range(x.ndim)]
    return x[slices]


def percent_change(old: Union[Tensor, float], new: Union[Tensor, float]) -> Union[Tensor, float]:
    r"""Computes percent change between two tensors or numbers.

    Args:
        old (:class:`torch.Tensor` or number):
            Previous value

        new (:class:`torch.Tensor` or number):
            New value

    Returns:
        Percent change (as a percentage).
    """
    return 100.0 * (new - old) / abs(old)


def percent_error_change(
    old: Union[Tensor, float], new: Union[Tensor, float], max_val: Union[Tensor, float] = 1.0
) -> Union[Tensor, float]:
    r"""Computes percent change in error two tensors or numbers.

    Args:
        old (:class:`torch.Tensor` or number):
            Previous value

        new (:class:`torch.Tensor` or number):
            New value

        max_val (:class:`torch.Tensor` or number):
            Maximum value for both ``old`` and ``new``. This is used to compute
            error as ``max_val - old`` or ``max_val - new``.

    Returns:
        Percent change in error (as a percentage).
    """
    old_err = max_val - old
    new_err = max_val - new
    return percent_change(old_err, new_err)
