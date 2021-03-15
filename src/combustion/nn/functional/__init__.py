#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from packaging import version

from ..activations.hsigmoid import hard_sigmoid
from ..activations.swish import hard_swish, swish
from ..loss.ciou import complete_iou_loss
from ..modules.dynamic_pad import patch_dynamic_same_pad
from .clamp_normalize import clamp_normalize
from .fill_masked import fill_normal
from .polar import cartesian_to_polar, polar_to_cartesian


if version.parse(torch.__version__) > version.parse("1.7.1"):
    from .fourier_conv import fourier_conv2d
else:

    def fourier_conv2d(*args, **kwargs):
        raise RuntimeError(f"fourier_conv2d requires torch>=1.8, but found {torch.__version__}")


__all__ = [
    "clamp_normalize",
    "complete_iou_loss",
    "swish",
    "hard_swish",
    "hard_sigmoid",
    "patch_dynamic_same_pad",
    "fill_normal",
    "fourier_conv2d",
    "cartesian_to_polar",
    "polar_to_cartesian",
]
