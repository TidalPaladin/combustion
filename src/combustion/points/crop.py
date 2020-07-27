#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


@torch.jit.script
def center_crop(coords: Tensor, size: Tuple[Optional[float], Optional[float], Optional[float]]) -> Tensor:
    r"""Crops a point cloud to a given size about the origin.
    See :class:`combustion.points.CenterCrop` for more details.
    """
    bounds = torch.empty(3).type_as(coords).float()
    for i, dim in enumerate(size):
        if dim is None:
            bounds[i] = float("inf")
        else:
            bounds[i] = dim / 2
    return torch.le(coords.abs(), bounds).all(dim=-1)


class CenterCrop(nn.Module):
    r"""Crops a point cloud to a given size about the origin.

    For a given dimension, included points :math:`p_i` will be calculated based on size
    :math:`s_d` in dimension :math:`d` as

    .. math::
        P' = \bigg\{(x, y, z) \in P \Big\vert \left\vert(x, y, z)\right\vert \leq \frac{(s_x, s_y, s_z)}{2}\bigg\}


    Args:

        size (tuple of optional floats):
            Cropped size along the x, y, and z axis respectively. A size can also be ``None`` in which case
            no cropping will be performed along that dimension.

    Shape
        * ``coords`` - :math:`(B, N, 3)` or :math:`(N, 3)`
        * Output - :math:`(B, N)` or :math:`(N)` depending on shape of ``coords``

    """

    def __init__(self, size: Tuple[int, int]):
        super().__init__()
        self.size = size

    def extra_repr(self):
        s = f"size={self.size}"
        return s

    def forward(self, coords: Tensor) -> Tensor:
        return center_crop(coords, self.size)
