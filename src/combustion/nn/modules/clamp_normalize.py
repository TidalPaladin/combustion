#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.nn as nn
from torch import Tensor


class ClampAndNormalize(nn.Module):
    r"""Clamps an input tensor and normalizes the clamped result to the range :math:`[0, 1]`.

    Args:

        minimum (float):
            The lower bound for clamping

        maximum (float):
            The upper bound for clamping
    """

    def __init__(self, minimum: float, maximum: float):
        super().__init__()
        minimum = float(minimum)
        maximum = float(maximum)
        if minimum >= maximum:
            raise ValueError(f"Expected minimum < maximum but found {minimum}, {maximum}")
        self._minimum = minimum
        self._maximum = maximum
        self._delta = maximum - minimum

    def extra_repr(self):
        s = f"min={self._minimum}, max={self._maximum}"
        return s

    def forward(self, inputs: Tensor) -> Tensor:
        inputs = inputs.float()
        outputs = inputs.clamp(self._minimum, self._maximum).sub_(self._minimum).div_(self._delta)
        return outputs
