#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Union

from numpy import ndarray
from torch import Tensor

from .bbox import _check_input


def to_8bit(img: Union[Tensor, ndarray], per_channel: bool = True) -> Tensor:
    r"""Converts an image Tensor or numpy array with an arbitrary range
    of values to a uint8 (byte) Tensor / numpy array. This is particularly useful
    when attempting to visualize images that have been standardized to zero mean
    unit variance or have higher than 8 bits of resolution.

    Args:
        img (Tensor or ndarray): The image to convert
        per_channel (bool): If true, quantize each channel separately

    Returns: Tensor
    """
    return_tensor = isinstance(img, Tensor)
    img: Tensor = _check_input(img, "img", ndim=(2, 4))
    if not isinstance(per_channel, bool):
        raise TypeError(f"Expected bool for per_channel, found {type(per_channel)}")

    # compute min/max/range
    if per_channel:
        minimum = img.min(dim=-1).values.min(dim=-1, keepdim=True).values
        maximum = img.max(dim=-1).values.max(dim=-1, keepdim=True).values
    else:
        minimum, maximum = img.min(), img.max()
    delta = maximum - minimum

    # map image to range 0-255
    original_shape = img.shape
    img = img.view(*img.shape[:-2], -1)
    img = (img + minimum.abs()) / (delta) * 255
    img = img.view(*original_shape).byte()

    if return_tensor:
        return img
    else:
        return img.numpy()
