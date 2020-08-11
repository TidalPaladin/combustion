#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .crop import CenterCrop, center_crop
from .transforms import RandomRotate, Rotate, random_rotate, rotate


# torch-scatter doesn't install correctly unless combustion[points] is installed after combustion
try:
    from .projection import projection_mapping, projection_mask
except ModuleNotFoundError:

    def projection_mask(*args, **kwargs):
        raise ImportError("Operation requires torch_scatter, please install it with `pip install combustion[points]`")

    def projection_mapping(*args, **kwargs):
        raise ImportError("Operation requires torch_scatter, please install it with `pip install combustion[points]`")


__all__ = [
    "Rotate",
    "rotate",
    "random_rotate",
    "RandomRotate",
    "center_crop",
    "CenterCrop",
    "projection_mask",
    "projection_mapping",
]
