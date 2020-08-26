#!/usr/bin/env python
# -*- coding: utf-8 -*-

from math import cos, radians, sin
from typing import Iterable, Tuple

import torch
import torch.nn as nn
from torch import Tensor


@torch.jit.script
def rotate(
    coords: Tensor, x: float = 0.0, y: float = 0.0, z: float = 0.0, degrees: bool = False, return_matrix: bool = False
) -> Tensor:
    # validate inputs
    if coords.ndim > 3 or coords.ndim < 2:
        raise ValueError(f"Expected 2 <= coords.ndim <= 3 but coords.ndim == {coords.ndim}")
    if coords.shape[-1] != 3:
        raise ValueError(f"Expected coords.shape[-1] == 3 but found coords.shape[-1] == {coords.shape[-1]}")
    x, y, z = float(x), float(y), float(z)

    # add batch dim if not present
    original_shape = coords.shape
    coords = coords.view(-1, coords.shape[-2], coords.shape[-1])
    if not coords.is_floating_point():
        coords = coords.float()
    output = torch.empty_like(coords)

    # degrees to radians if desired
    if degrees:
        x = radians(x)
        y = radians(y)
        z = radians(z)

    # build rotation matrices
    rot_x = torch.tensor([[1.0, 0.0, 0.0], [0.0, cos(x), -sin(x)], [0.0, sin(x), cos(x)]], device=coords.device)
    rot_y = torch.tensor([[cos(y), 0.0, sin(y)], [0.0, 1.0, 0.0], [-sin(y), 0.0, cos(y)]], device=coords.device)
    rot_z = torch.tensor([[cos(z), -sin(z), 0.0], [sin(z), cos(z), 0.0], [0.0, 0.0, 1.0]], device=coords.device)
    rotation_matrix = torch.chain_matmul(rot_z, rot_x, rot_y).unsqueeze_(0).type_as(coords)
    assert rotation_matrix.ndim == 3
    assert rotation_matrix.size() == torch.Size((1, 3, 3))

    if return_matrix:
        return rotation_matrix

    # perform rotation
    torch.bmm(coords, rotation_matrix, out=output)
    output = output.view(original_shape)

    return output


class Rotate(nn.Module):
    r"""Rotates a collection of points using rotation values in radians or degrees.

    Args:

        x (float):
            Rotation about x-axis

        y (float):
            Rotation about y-axis

        z (float):
            Rotation about z-axis

        degrees (bool):
            By default rotations are in radians. When ``degrees=True``, rotations are treated as degrees.

    Shape
        * ``coords`` - :math:`(B, N, 3)` or :math:`(N, 3)`
        * Output - same as ``coords``
    """

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0, degrees: bool = False):
        super().__init__()
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.degrees = degrees

    def extra_repr(self):
        s = f"x={self.x}, y={self.y}, z={self.z}"
        if self.degrees:
            s += ", degrees=True"
        return s

    def forward(self, coords: Tensor) -> Tensor:
        return rotate(coords, self.x, self.y, self.z, self.degrees)


def random_rotate(
    coords: Tensor,
    x: Tuple[float, float] = (0.0, 0.0),
    y: Tuple[float, float] = (0.0, 0.0),
    z: Tuple[float, float] = (0.0, 0.0),
    degrees: bool = False,
    return_matrix: bool = False,
) -> Tensor:
    for var, s in zip((x, y, z), ("x", "y", "z")):
        if not isinstance(var, Iterable):
            raise TypeError(f"Expected {s} to be iterable, but found {type(var)}")
        if len(var) != 2:
            raise ValueError(f"Expected {s} to be of length 2, but found {len(var)}")
        if var[1] < var[0]:
            raise ValueError(f"Expected {s}_low <= {s}_high, but found {(var[0], var[1])}")

    # generate random rotation
    _ = torch.tensor([[x[0], x[1]], [y[0], y[1]], [z[0], z[1]]]).type_as(coords).float()
    lows = _.min(dim=-1).values
    highs = _.max(dim=-1).values
    rots = torch.rand_like(highs)
    rots.mul_(highs - lows).add_(lows)

    return rotate(coords, rots[0], rots[1], rots[2], degrees, return_matrix)


class RandomRotate(nn.Module):
    r"""Rotates a collection of points randomly between a minimum and maximum possible rotation.

    Args:

        x (tuple of floats):
            Minimum and maximum rotation about x-axis.

        y (tuple of floats):
            Minimum and maximum rotation about y-axis.

        z (tuple of floats):
            Minimum and maximum rotation about z-axis.

        degrees (bool):
            By default rotations are in radians. When ``degrees=True``, rotations are treated as degrees.

    Shape
        * ``coords`` - :math:`(B, N, 3)` or :math:`(N, 3)`
        * Output - same as ``coords``
    """

    def __init__(
        self,
        x: Tuple[float, float] = (0.0, 0.0),
        y: Tuple[float, float] = (0.0, 0.0),
        z: Tuple[float, float] = (0.0, 0.0),
        degrees: bool = False,
    ):
        super().__init__()
        for var, s in zip((x, y, z), ("x", "y", "z")):
            if not isinstance(var, Iterable):
                raise TypeError(f"Expected {s} to be iterable, but found {type(var)}")
            if len(var) != 2:
                raise ValueError(f"Expected {s} to be of length 2, but found {len(var)}")
            if var[1] < var[0]:
                raise ValueError(f"Expected {s}_low <= {s}_high, but found {(var[0], var[1])}")

        self.x = x
        self.y = y
        self.z = z
        self.degrees = degrees

    def extra_repr(self):
        s = f"x={self.x}, y={self.y}, z={self.z}"
        if self.degrees:
            s += ", degrees=True"
        return s

    def forward(self, coords: Tensor) -> Tensor:
        return random_rotate(coords, self.x, self.y, self.z, self.degrees)


@torch.jit.script
def center(coords: Tensor, inplace: bool = False, strategy: str = "minmax") -> Tensor:
    r"""Centers a collection of points about the origin based on their Cartesian coordinates.

    Args:
        coords (:class:`torch.Tensor`):
            Cartesian coordinates of the input points.

        inplace (bool):
            If true, perform the operation inplace. Inplace operation is only possible when
            ``coords`` is a floating point type.

        strategy (str):
            - ``'minmax'`` - center based on the range of points for each dimension
            - ``'mean'`` - center such that the mean of points for each dimension is zero

    Shape
        * ``coords`` - :math:`(N, P)`
        * Output - same as input

    """
    # validate inputs
    if coords.ndim > 3 or coords.ndim < 2:
        raise ValueError(f"Expected 2 <= coords.ndim <= 3 but coords.ndim == {coords.ndim}")

    if not coords.is_floating_point():
        if inplace:
            raise RuntimeError("Inplace not possible when coords is non-floating point")
        coords = coords.float()

    if strategy == "minmax":
        mins = coords.min(dim=0).values
        maxes = coords.max(dim=0).values
        offset = (maxes - mins).div_(2).add_(mins)
    elif strategy == "mean":
        offset = coords.mean(dim=0)
    else:
        raise ValueError(f"Unknown strategy {strategy}")

    if inplace:
        return coords.sub_(offset)
    else:
        return coords.sub(offset)
