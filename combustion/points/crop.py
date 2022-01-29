#!/usr/bin/env python
# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
from torch import Tensor


@torch.jit.script
def center_crop(
    coords: Tensor, crop_x: float = float("inf"), crop_y: float = float("inf"), crop_z: float = float("inf")
) -> Tensor:
    r"""Crops a point cloud to a given crop about the origin.
    See :class:`combustion.points.CenterCrop` for more details.
    """
    bounds = torch.tensor([crop_x, crop_y, crop_z], device=coords.device).float().div_(2)
    return torch.le(coords.abs(), bounds).all(dim=-1)


class CenterCrop(nn.Module):
    r"""Crops a point cloud to a given crop about the origin.

    For a given dimension, included points :math:`p_i` will be calculated based on crop
    :math:`s_d` in dimension :math:`d` as

    .. math::
        P' = \bigg\{(x, y, z) \in P \Big\vert \left\vert(x, y, z)\right\vert \leq \frac{(s_x, s_y, s_z)}{2}\bigg\}


    Args:

        crop (tuple of optional floats):
            Cropped crop along the x, y, and z axis respectively. A crop can also be ``None`` in which case
            no cropping will be performed along that dimension.

    Shape
        * ``coords`` - :math:`(B, N, 3)` or :math:`(N, 3)`
        * Output - :math:`(B, N)` or :math:`(N)` depending on shape of ``coords``

    """

    def __init__(self, crop_x: float = float("inf"), crop_y: float = float("inf"), crop_z: float = float("inf")):
        super().__init__()
        self.crop_x = crop_x
        self.crop_y = crop_y
        self.crop_z = crop_z

    def extra_repr(self):
        s = f"crop={(self.crop_x, self.crop_y, self.crop_z)}"
        return s

    def forward(self, coords: Tensor) -> Tensor:
        return center_crop(coords, self.crop_x, self.crop_y, self.crop_z)
