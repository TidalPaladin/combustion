#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.nn as nn
from torch import Tensor


class ClampAndNormalize(nn.Module):
    r"""Clamps an input tensor and normalizes the clamped result to a fixed range.

    Args:

        minimum (float):
            The lower bound for clamping

        maximum (float):
            The upper bound for clamping

        norm_min (float):
            The lower bound of the normalized range

        norm_max (float):
            The upper bound of the normalized range
    """

    def __init__(self, minimum: float, maximum: float, norm_min: float = 0.0, norm_max: float = 1.0):
        super().__init__()
        minimum = float(minimum)
        maximum = float(maximum)
        if minimum >= maximum:
            raise ValueError(f"Expected minimum < maximum but found {minimum}, {maximum}")

        norm_min = float(norm_min)
        norm_max = float(norm_max)
        if norm_min >= norm_max:
            raise ValueError(f"Expected norm_min < norm_max but found {norm_min}, {norm_max}")

        self._minimum = minimum
        self._maximum = maximum
        self._norm_min = norm_min
        self._norm_max = norm_max
        self._delta = maximum - minimum
        self._output_delta = norm_max - norm_min

    def extra_repr(self):
        s = f"min={self._minimum}, max={self._maximum}"
        if self._norm_min != 0:
            s += f", norm_min={self._norm_min}"
        if self._norm_max != 1:
            s += f", norm_max={self._norm_max}"
        return s

    def forward(self, inputs: Tensor) -> Tensor:
        inputs = inputs.float()
        outputs = inputs.clamp(self._minimum, self._maximum).sub_(self._minimum).div_(self._delta)

        if self._norm_min != 0 or self._norm_max != 1:
            outputs = outputs.mul_(self._output_delta).add_(self._norm_min)

        return outputs
