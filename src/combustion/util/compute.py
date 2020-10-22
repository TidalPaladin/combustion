#!/usr/bin/env python
# -*- coding: utf-8 -*-

from numbers import Number
from typing import Union

from torch import Tensor


def percent_change(old: Union[Tensor, Number], new: Union[Tensor, Number]) -> Union[Tensor, Number]:
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
    old: Union[Tensor, Number], new: Union[Tensor, Number], max_val: Union[Tensor, Number] = 1.0
) -> Union[Tensor, Number]:
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
