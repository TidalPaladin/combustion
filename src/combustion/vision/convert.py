#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Union

import torch
from numpy import ndarray
from torch import Tensor

from combustion.util import check_is_array


def to_8bit(img: Union[Tensor, ndarray], per_channel: bool = True, same_on_batch: bool = False) -> Tensor:
    r"""Converts an image Tensor or numpy array with an arbitrary range
    of values to a uint8 (byte) Tensor / numpy array. This is particularly useful
    when attempting to visualize images that have been standardized to zero mean
    unit variance or have higher than 8 bits of resolution.

    Args:
        img (Tensor or ndarray): The image to convert
        per_channel (bool, optional): If true, quantize each channel separately
        same_on_batch (bool, optional): If true, use batch-wide minima/maxima for quantization

    Shape:
        - Image: :math:`(C, H, W)` or :math:`(N, C, H, W)` where :math:`N` is an optional batch
          dimension.
    """
    return_tensor = isinstance(img, Tensor)
    check_is_array(img, "img")
    img: Tensor = torch.as_tensor(img)
    if not isinstance(per_channel, bool):
        raise TypeError(f"Expected bool for per_channel, found {type(per_channel)}")
    if not isinstance(same_on_batch, bool):
        raise TypeError(f"Expected bool for same_on_batch, found {type(same_on_batch)}")

    # compute min/max/range
    minimum = img.min(dim=-1).values.min(dim=-1, keepdim=True).values
    maximum = img.max(dim=-1).values.max(dim=-1, keepdim=True).values

    if not per_channel:
        minimum = minimum.min(dim=-2, keepdim=True).values
        maximum = maximum.max(dim=-2, keepdim=True).values

    if same_on_batch and img.ndim == 4:
        minimum = minimum.min(dim=0, keepdim=True).values
        maximum = maximum.max(dim=0, keepdim=True).values

    delta = maximum - minimum

    # map image to range 0-255
    original_shape = img.shape
    img = img.view(*img.shape[:-2], -1).float().sub(minimum).mul_(255).div_(delta).round_().view(*original_shape).byte()

    if return_tensor:
        return img
    else:
        return img.numpy()
