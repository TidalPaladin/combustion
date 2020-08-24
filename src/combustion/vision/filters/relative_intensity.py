#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List, Tuple

import torch
from torch import Tensor

from combustion.vision.filters import GaussianBlur2d, gaussian_blur2d


@torch.jit.script
def _combine_baselines(baselines: List[Tensor], combine: str) -> Tensor:
    final_baseline = baselines[0].neg()
    for baseline in baselines[1:]:
        baseline = baseline.neg()
        if combine == "max":
            final_baseline = torch.max(final_baseline, baseline)
        elif combine == "min":
            final_baseline = torch.min(final_baseline, baseline)
        else:
            final_baseline = torch.add(final_baseline, baseline)

    if combine == "mean":
        final_baseline.div_(len(baselines))
    return final_baseline.neg_()


def relative_intensity(
    inputs: Tensor,
    kernel_size: List[Tuple[int, int]],
    sigma: List[Tuple[int, int]],
    border_type: str = "reflect",
    combine: str = "max",
) -> Tensor:
    r"""Computes relative intensity of a 2D input by subtracting the input from gaussian blur of the input.
    This operation attenuates low frequency components of the input.

    .. note::
        This function performs convolution via a multiplication in the frequency domain, making it efficient
        for large kernel sizes.

    Args:
        inputs (:class:`torch.Tensor`):
            The input to transform

        kernel_size (:class:`List[Tuple[int, int]]`):
            List of kernel sizes for the Gaussian blur

        sigma (:class:`List[Tuple[int, int]]`):
            List of sigmas for the Gaussian blur

        border_type (str):
            ``constant``, ``reflect``, ``replicate``, or ``circular``. See :class:`kornia.filters.GaussianBlur2d`.

        combine (str):
            How to combine outputs for multiple Gaussian kernels. One of
            ``max``, ``min``, ``mean``, ``sum``.

    Shapes
        * ``inputs`` - :math:`(N, C, H, W)`
        * Output - same as input
    """
    assert len(kernel_size)
    assert len(kernel_size) == len(sigma)
    assert isinstance(kernel_size[0], Tuple)
    baselines: List[Tensor] = []
    for kernel, sig in zip(kernel_size, sigma):
        baseline: Tensor = gaussian_blur2d(inputs, kernel, sig, border_type)
        baselines.append(baseline)

    final_baseline = _combine_baselines(baselines, combine)
    if combine == "sum":
        inputs = inputs * len(baselines)
    return inputs - final_baseline


class RelativeIntensity:
    r"""See :class:`combustion.vision.filters.relative_intensity`."""

    def __init__(
        self,
        kernel_size: List[Tuple[int, int]],
        sigma: List[Tuple[int, int]],
        border_type: str = "reflect",
        combine: str = "max",
    ):
        self._kernel_size = kernel_size
        self._sigma = sigma
        self._border_type = border_type
        self.combine = combine
        self.blurs = [GaussianBlur2d(kernel, sig, border_type) for kernel, sig in zip(kernel_size, sigma)]

    def __call__(self, inputs: Tensor):
        baselines: List[Tensor] = []
        for blur in self.blurs:
            baseline = blur(inputs)
            baselines.append(baseline)

        final_baseline = _combine_baselines(baselines, self.combine)
        if self.combine == "sum":
            inputs = inputs * len(baselines)
        return inputs - final_baseline
