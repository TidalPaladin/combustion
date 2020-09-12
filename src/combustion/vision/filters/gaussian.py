#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Tuple, Union

import torch.nn as nn
from kornia.filters import get_gaussian_kernel2d
from torch import Tensor

from combustion.nn.functional import fourier_conv2d


class GaussianBlur2d(nn.Module):
    r"""Implements Gaussian blurring via multiplication in the frequency domain.
    This method is more efficient for large kernel sizes.

    Args:
        kernel_size (tuple of ints):
            Size of gaussian kernel

        sigma (tuple of ints):
            Sigma for gaussian kernel

        padding_mode (str):
            See :func:`torch.nn.functional.pad`

        fill_value (float):
            See :func:`torch.nn.functional.pad`

    Shape:
        * Input - :math:`(N, C, H, W)`
        * Output - Same as input
    """

    def __init__(
        self, kernel_size: Tuple[int, int], sigma: Tuple[int, int], padding_mode: str = "reflect", fill_value: float = 0
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.padding_mode = padding_mode
        self.fill_value = fill_value
        self._kernel = get_gaussian_kernel2d(kernel_size, sigma).unsqueeze_(0)

    def extra_repr(self) -> str:
        s = f"kernel_size={self.kernel_size}, sigma={self.sigma}, padding_mode={self.padding_mode}"
        if self.padding_mode == "constant":
            s += ", fill_value={self.fill_value}"
        return s

    def forward(self, inputs: Tensor) -> Tensor:
        kernel = self._kernel.type_as(inputs)
        height, width = inputs.shape[-2:]
        kernel_h, kernel_w = kernel.shape[-2:]
        padding = (kernel_h // 2, kernel_w // 2)
        return fourier_conv2d(
            inputs, kernel, padding=padding, padding_mode=self.padding_mode, fill_value=self.fill_value
        )


def gaussian_blur2d(
    inputs: Tensor,
    kernel_size: Union[int, Tuple[int, int]],
    sigma: Union[int, Tuple[int, int]],
    padding_mode: str = "reflect",
    fill_value: float = 0.0,
) -> Tensor:
    r"""See :class:`combustion.vision.filters.GaussianBlur2d`"""
    return GaussianBlur2d(kernel_size, sigma, padding_mode, fill_value)(inputs)
