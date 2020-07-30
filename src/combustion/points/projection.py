#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional, Tuple

import torch
import torch_scatter
from torch import Tensor

from combustion.points import center_crop


@torch.jit.script
def projection_mask(coords: Tensor, resolution: float = 1.0, image_size: Optional[Tuple[int, int]] = None) -> Tensor:
    r"""Performs a parallel projection of Cartesian coordinates at a given resolution, computing a 2D grid of indices
    that give the 3D coordinates mapped to each 2D pixel. Batched operation is supported, however the inferred size
    of the 2D projection will be that of the largest batch element.

    .. note::
        Pixels in the 2D projection that did not have a point mapped to them will be assigned value ``-1``.

    .. warning::
        For an undetermined reason, this function seems to have a greater runtime on GPU tensors than CPU tensors.

    Args:

        coords (Tensor):
            Cartesian coordinates comprising a point cloud, in XYZ order.

        resolution (float):
            The resolution at which to construct the 2D projection

        image_size (optional, 2-tuple of ints):
            By default, the size of the resultant projection is determined based on ``resolution`` and the
            range of x-y coordinates in ``coords``. An optional image size can be provided to override
            this behavior.

    Shapes
        * ``coords`` - :math:`(N, 3)` or :math:`(B, N, 3)`
        * Output - :math:`(H, W)`

    """
    # apply batch dim if not present
    if coords.ndim < 3:
        coords = coords.view(1, coords.shape[-2], coords.shape[-1])

    mask = torch.empty(1).type_as(coords)
    x, y, z = coords[..., 0], coords[..., 1], coords[..., 2]

    # crop point cloud if image_size is given
    if image_size is not None:
        height, width = image_size
        mask = center_crop(coords, (height * resolution, width * resolution, None))
        assert mask.any()
        coords = coords[mask]
        x, y, z = coords[..., 0], coords[..., 1], coords[..., 2]

    # shift xy coords to the origin
    real_mins = coords[..., :2].min(dim=-2).values
    coords[..., :2].sub_(real_mins)
    mins = torch.zeros_like(real_mins)

    # get min/max xy coords
    maxes = coords[..., :2].max(dim=-2).values
    min_x, min_y = mins[..., 0], mins[..., 1]
    max_x, max_y = maxes[..., 0], maxes[..., 1]

    # compute height/width of final grid
    if image_size is None:
        height = (max_y - min_y).floor_divide_(resolution).max().item()
        width = (max_x - min_x).floor_divide_(resolution).min().item()
        height = int(height)
        width = int(width)
    else:
        height, width = image_size
    assert height > 0
    assert width > 0

    # map each point to a height/width in the 2d grid
    x, y, z = coords[..., 0], coords[..., 1], coords[..., 2]
    h_coords = y.floor_divide(resolution).clamp_max_(height - 1)
    w_coords = x.floor_divide(resolution).clamp_max_(width - 1)

    # convert x,y coords into a single flat index
    flat_coords = h_coords.mul_(width).add_(w_coords).long()

    # get the closest (in z-axis) point within each xy grid location
    _, unoccluded_points = torch_scatter.scatter_max(z, flat_coords, dim_size=height * width)

    # undo the inplace shift to origin
    coords[..., :2].add_(real_mins)

    # assign -1 to pixels that didn't contain any point
    no_data = unoccluded_points == flat_coords.numel()
    unoccluded_points[no_data] = torch.tensor(-1).type_as(unoccluded_points)

    # if image_size was given we may have cropped away some points from original coords
    # -> mask must be adjusted to index points in the raw point cloud, not the cropped cloud
    if image_size is not None:
        unoccluded_points = mask.nonzero().flatten()[unoccluded_points]
        unoccluded_points[no_data] = torch.tensor(-1).type_as(unoccluded_points)

    _ = unoccluded_points.view(height, width)
    return _
