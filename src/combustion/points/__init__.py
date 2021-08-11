#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Tuple

from torch import Tensor

from .crop import CenterCrop, center_crop
from .transforms import RandomRotate, Rotate, center, random_rotate, rotate


# torch-scatter doesn't install correctly unless combustion[points] is installed after combustion
try:
    import torch_scatter  # type: ignore
except ImportError:
    torch_scatter = None

if torch_scatter is not None:
    from .projection import projection_mapping, projection_mask
else:

    def projection_mask(*args, **kwargs) -> Tensor:  # type: ignore
        raise ImportError("Operation requires torch_scatter, please install it with `pip install combustion[points]`")

    def projection_mapping(*args, **kwargs) -> Tuple[Tensor, Tensor, Tuple[int, int]]:  # type: ignore
        raise ImportError("Operation requires torch_scatter, please install it with `pip install combustion[points]`")


__all__ = [
    "center",
    "Rotate",
    "rotate",
    "random_rotate",
    "RandomRotate",
    "center_crop",
    "CenterCrop",
    "projection_mask",
    "projection_mapping",
]
