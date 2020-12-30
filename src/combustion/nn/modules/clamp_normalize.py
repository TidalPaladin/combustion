#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.nn as nn
from torch import Tensor

from ..functional.clamp_normalize import clamp_normalize


class ClampAndNormalize(nn.Module):
    r"""Clamps an input tensor and normalizes the clamped result to a fixed range.
    When called with default arguments, this operation is equivalent to min-max
    normalization to the range :math:`[0, 1]`.

    Args:

        minimum (float):
            The lower bound for clamping.

        maximum (float):
            The upper bound for clamping

        norm_min (float):
            The lower bound of the normalized range

        norm_max (float):
            The upper bound of the normalized range

        inplace (bool):
            Whether or not to perform the operation in-place
    """

    def __init__(
        self,
        minimum: float = float("-inf"),
        maximum: float = float("inf"),
        norm_min: float = 0.0,
        norm_max: float = 1.0,
        inplace: bool = False,
    ):
        super().__init__()
        minimum = float(minimum)
        maximum = float(maximum)
        if minimum >= maximum:
            raise ValueError(f"Expected minimum < maximum but found {minimum}, {maximum}")

        norm_min = float(norm_min)
        norm_max = float(norm_max)
        if norm_min >= norm_max:
            raise ValueError(f"Expected norm_min < norm_max but found {norm_min}, {norm_max}")

        self.minimum = minimum
        self.maximum = maximum
        self.norm_min = norm_min
        self.norm_max = norm_max
        self.inplace = bool(inplace)

    def extra_repr(self):
        s = f"min={self.minimum}, max={self.maximum}"
        if self.minimum != float("-inf"):
            s += f", norm_min={self.norm_min}"
        if self.norm_min != 0:
            s += f", norm_min={self.norm_min}"
        if self.norm_max != 1:
            s += f", norm_max={self.norm_max}"
        if self.inplace:
            s += f", inplace={inplace}"
        return s

    def forward(self, inputs: Tensor) -> Tensor:
        return clamp_normalize(inputs, self.minimum, self.maximum, self.norm_min, self.norm_max)
