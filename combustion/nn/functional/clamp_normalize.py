#!/usr/bin/env python
# -*- coding: utf-8 -*-


import torch
from torch import Tensor


@torch.jit.script
def clamp_normalize(
    inputs: Tensor,
    minimum: float = float("-inf"),
    maximum: float = float("inf"),
    norm_min: float = 0.0,
    norm_max: float = 1.0,
    inplace: bool = True,
    eps: float = 1e-8,
) -> Tensor:
    if maximum <= minimum:
        raise ValueError(f"Expected maximum > minimum: got {maximum} vs {minimum}")
    if norm_max <= norm_min:
        raise ValueError(f"Expected norm_max > norm_min: got {norm_max} vs {norm_min}")
    if eps <= 0:
        raise ValueError(f"Expected eps > 0: got {eps}")

    if not inputs.is_floating_point():
        inputs = inputs.float()

    minimum = float(minimum) if minimum != float("-inf") else float(inputs.amin().item())
    maximum = float(maximum) if maximum != float("inf") else float(inputs.amax().item())

    delta = max(maximum - minimum, eps)
    output_delta = norm_max - norm_min
    assert output_delta > 0

    if inplace:
        outputs = inputs.clamp_(minimum, maximum).sub_(minimum).div_(delta)
    else:
        outputs = inputs.clamp(minimum, maximum).sub(minimum).div(delta)

    # only apply renormalization if necessary
    if norm_min != 0 or norm_max != 1:
        if inplace:
            outputs = outputs.mul_(output_delta).add_(norm_min)
        else:
            outputs = outputs.mul(output_delta).add(norm_min)

    return outputs
