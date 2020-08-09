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
    that give the 3D coordinates mapped to each 2D pixel.

    .. note::
        Pixels in the 2D projection that did not have a point mapped to them will be assigned value ``-1``.

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
        * ``coords`` - :math:`(N, 3)` or :math:`(1, N, 3)`
        * Output - :math:`(H, W)`

    """
    # apply batch dim if not present
    coords = coords.view(coords.shape[-2], coords.shape[-1])

    mask = torch.empty(1, device=coords.device).type_as(coords)
    x, y, z = coords[..., 0], coords[..., 1], coords[..., 2]

    # find xy mins/maxes
    min_x = coords[..., 0].min()
    max_x = coords[..., 0].max()
    min_y = coords[..., 1].min()
    max_y = coords[..., 1].max()
    mins = torch.stack([min_x, min_y], dim=0)
    maxes = torch.stack([max_x, max_y], dim=0)

    # use given height/width or calculate one from min/max
    if image_size is not None:
        height, width = image_size
    else:
        height = int(max_y.sub(min_y).floor_divide_(resolution).long().item())
        width = int(max_x.sub(min_x).floor_divide_(resolution).long().item())
    assert height > 0
    assert width > 0

    # crop point cloud based on resolution and image size
    crop_height = float(height * resolution)
    crop_width = float(width * resolution)
    mask = center_crop(coords, (crop_height, crop_width, None))
    assert mask.any()
    coords = coords[mask]
    x, y, z = coords[..., 0], coords[..., 1], coords[..., 2]

    # map each point to a height/width in the 2d grid
    x, y, z = coords[..., 0], coords[..., 1], coords[..., 2]
    h_coords = y.sub(mins[1]).floor_divide_(resolution).clamp_max_(height - 1)
    w_coords = x.sub(mins[0]).floor_divide_(resolution).clamp_max_(width - 1)

    # convert x,y coords into a single flat index for scatter_max
    flat_coords = h_coords.mul_(width).add_(w_coords).long()

    # get the closest (in z-axis) point within each xy grid location
    _, unoccluded_points = torch_scatter.scatter_max(z, flat_coords, dim_size=height * width)

    # assign -1 to pixels that didn't contain any point
    no_data = unoccluded_points == flat_coords.numel()
    fill_no_data = torch.tensor(-1, device=unoccluded_points.device)
    unoccluded_points[no_data] = fill_no_data

    # if image_size was given we may have cropped away some points from original coords
    # -> mask must be adjusted to index points in the raw point cloud, not the cropped cloud
    unoccluded_points = mask.nonzero().flatten()[unoccluded_points]
    unoccluded_points[no_data] = fill_no_data

    _ = unoccluded_points.view(height, width)
    return _
