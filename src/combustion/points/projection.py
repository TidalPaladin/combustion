#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional, Tuple

import torch
import torch_scatter
from torch import Tensor


# @torch.jit.script
def projection_mask(coords: Tensor, resolution: float = 1.0, image_size: Optional[Tuple[int, int]] = None) -> Tensor:
    mask = torch.empty(1)
    # coords = coords.round()
    x, y, z = coords[..., 0], coords[..., 1], coords[..., 2]

    # crop point cloud
    if image_size is not None:
        height, width = image_size
        mask = center_crop_mask(coords, (height * resolution, width * resolution))
        assert mask.any()
        coords = coords[mask]
        x, y, z = coords[..., 0], coords[..., 1], coords[..., 2]

    # shift coords to the origin
    mins = coords.min(dim=0).values
    coords.sub_(mins)

    # get min/max xy coords
    maxes = coords[..., :2].max(dim=0).values
    min_x, min_y = 0, 0
    max_x, max_y = maxes[0], maxes[1]

    # compute height/width of grid
    if image_size is None:
        height = int((max_y - min_y) / resolution)
        width = int((max_x - min_x) / resolution)
    else:
        height, width = image_size

    # map each point to a 2d grid location
    x, y, z = coords[..., 0], coords[..., 1], coords[..., 2]

    h_coords = y.div(resolution).round_().clamp_max_(height - 1).long()
    w_coords = x.div(resolution).round_().clamp_max(width - 1).long()

    # convert x,y coords into a single coordinate for point_cloud indexing
    coords = h_coords * width + w_coords
    # assert coords.min() >= 0
    # assert coords.max() <= height * width, f"{coords.max()} vs {height * width}"

    # get the closest (in z-axis) point within each xy grid location
    _, unoccluded_points = torch_scatter.scatter_max(z, coords, dim_size=height * width)

    # handle pixels that didn't contain a point
    no_data = unoccluded_points == coords.numel()
    unoccluded_points[no_data] = torch.tensor(-1).type_as(unoccluded_points)

    if image_size is not None:
        unoccluded_points = mask.nonzero().flatten()[unoccluded_points]
        unoccluded_points[no_data] = torch.tensor(-1).type_as(unoccluded_points)

    _ = unoccluded_points.view(height, width)
    return _
