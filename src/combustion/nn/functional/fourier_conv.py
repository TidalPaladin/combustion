#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


def _check_mul_complex_inputs(a: Tensor, b: Tensor):
    if a.shape[-1] != 2:
        raise ValueError(f"Expected a.shape[-1] == 2, but found shape {a.shape}")
    if b.shape[-1] != 2:
        raise ValueError(f"Expected b.shape[-1] == 2, but found shape {b.shape}")
    if a.shape != b.shape:
        raise ValueError(f"Expected a.shape == b.shape, but found {a.shape}, {b.shape}")


def mul_complex(a: Tensor, b: Tensor) -> Tensor:
    _check_mul_complex_inputs(a, b)
    op = "bc...,dc...->bd..."
    t1 = torch.einsum(op, a[..., 0], b[..., 0]) + torch.einsum(op, a[..., 1], b[..., 1])
    t2 = torch.einsum(op, a[..., 1], b[..., 0]) - torch.einsum(op, a[..., 0], b[..., 1])
    return torch.stack([t1, t2], dim=-1)


@torch.jit.script
def mul_complex_(a: Tensor, b: Tensor) -> Tensor:
    _check_mul_complex_inputs(a, b)
    t1 = (a[..., 0] * b[..., 0]).add_(a[..., 1] * b[..., 1])
    t2 = (a[..., 1] * b[..., 0]).sub_(a[..., 0] * b[..., 1])
    a[..., 0] = t1
    a[..., 1] = t2
    return a


@torch.jit.script
def fourier_conv2d(
    inputs: Tensor,
    kernel: Tensor,
    stride: Tuple[int, int] = (1, 1),
    padding: Tuple[int, int] = (0, 0),
    padding_mode: str = "constant",
    fill_value: float = 0,
    bias: Optional[Tensor] = None,
) -> Tensor:
    r"""Performs 2D convolution as a multiplication in the frequency domain. This
    approach is more efficient for large kernel sizes and small strides.

    Args:
        inputs (:class:`torch.Tensor`):
            Tensor to convolve over

        kernel (:class:`torch.Tensor`):
            Convolution kernel

        stride (tuple of int, int):
            Convolution stride

        padding (tuple of int, int):
            Padding to be applied to inputs

        padding_mode (str):
            Same as :func:`torch.nn.functional.pad` ``padding_mode``

        fill_value (float):
            Same as :func:`torch.nn.functional.pad` ``fill_value``

        bias (:class:`torch.Tensor`):
            Bias term

    Shape
        * ``input`` - :math:`(N, C, H_i, W_i)`
        * ``kernel`` - :math:`(N, C, H_k, W_k)`
        * ``bias`` - :math:`(C)`
    """
    original_shape = inputs.shape
    spatial_ndim = 2

    # expand kernel.ndim to match inputs.ndim
    while kernel.ndim < inputs.ndim:
        kernel.unsqueeze_(0)
    assert kernel.ndim == inputs.ndim

    # pad inputs and fft
    _pad = (padding[1], padding[1], padding[0], padding[0])
    inputs = F.pad(inputs, _pad, padding_mode, fill_value)
    fft_inputs = torch.rfft(inputs, spatial_ndim)

    # get padding to place kernel in tensor of shape input.shape
    kernel_padding: List[int] = []
    for i in range(spatial_ndim):
        size = inputs.shape[-(i + 1)] - kernel.shape[-(i + 1)]
        assert size >= 0
        kernel_padding += [0, size]
    assert len(kernel_padding) == 2 * spatial_ndim

    # pad kernel and fft
    _ = F.pad(kernel, kernel_padding)
    fft_kernel = torch.rfft(_, spatial_ndim)

    # do convolution as multiplication in frequency domain and inverse fourier
    result = mul_complex(fft_inputs, fft_kernel).irfft(spatial_ndim, signal_sizes=inputs.shape[2:])
    result = result[..., :: stride[0], :: stride[1]]

    # crop to match expected output shape
    crop: List[int] = []
    for i, (s, p) in enumerate(zip(stride, padding)):
        l = original_shape[2 + i]
        k = kernel.shape[2 + i]
        crop.append((l - k + 2 * p) // s + 1)
    result = result[..., : crop[0], : crop[1]]
    result = result.contiguous()

    if bias is not None:
        result += bias

    return result
