#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import torch_scatter
from torch import Tensor

from combustion.points import center_crop


@torch.jit.script
def projection_mapping(
    coords: Tensor,
    resolution: float = 1.0,
    image_size: Optional[Tuple[int, int]] = None,
) -> Tuple[Tensor, Tensor, Tuple[int, int]]:
    r"""Performs a parallel projection of Cartesian coordinates at a given resolution, computing a mapping of each
    3D point to a 2D pixel. Returns a mapping of points within the image area to height and witdth pixel coordinates,
    along with a mask of points that lie within the projected window and the height and width of the 2D projection.

    Args:

        coords (Tensor):
            Cartesian coordinates comprising a point cloud, in XYZ order.

        resolution (float):
            The resolution at which to construct the 2D projection

        image_size (optional, 2-tuple of ints):
            By default, the size of the resultant projection is determined based on ``resolution`` and the
            range of x-y coordinates in ``coords``. An optional image size can be provided to override
            this behavior.

    Returns:
        Tuple of the form ``mapping, mask (height, width)``

    Shapes
        * ``coords`` - :math:`(N, 3)` or :math:`(1, N, 3)`
        * Mapping - :math:`(N, 2)` where the final dimension gives the height and width coordinate
        * Mask - :math:`(N)`
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

    if image_size is not None:
        height, width = image_size

        # crop point cloud based on resolution and image size
        crop_height = float(height)
        crop_width = float(width)
        mask = center_crop(coords, crop_height, crop_width)
        assert mask.any()
        coords = coords[mask]

    else:
        # calculate size one from min/max
        height = int(max_y.sub(min_y).floor_divide_(resolution).long().item())
        width = int(max_x.sub(min_x).floor_divide_(resolution).long().item())
        mask = torch.tensor([True], device=coords.device).expand(coords.shape[0])

    assert height > 0
    assert width > 0

    x, y, z = coords[..., 0], coords[..., 1], coords[..., 2]

    # recompute mins/maxes after cropping
    mins = torch.min(coords[..., :2], dim=0).values
    maxes = torch.max(coords[..., :2], dim=0).values

    # map each point to a height/width in the 2d grid
    x, y, z = coords[..., 0], coords[..., 1], coords[..., 2]

    mapping = coords[..., :2].roll(1, dims=-1).sub(mins.roll(1, dims=-1)).floor_divide(resolution)
    mapping[..., 0].clamp_max_(height - 1)
    mapping[..., 1].clamp_max_(width - 1)

    return mapping, mask, (height, width)


@torch.jit.script
def projection_mask(
    coords: Tensor,
    resolution: float = 1.0,
    image_size: Optional[Tuple[int, int]] = None,
    padding_mode: str = "constant",
    fill_value: float = -1,
    projection_map: Optional[Tuple[Tensor, Tensor, Tuple[int, int]]] = None,
) -> Tensor:
    r"""Performs a parallel projection of Cartesian coordinates at a given resolution, computing a 2D grid of indices
    that give the 3D coordinates mapped to each 2D pixel.

    Args:

        coords (Tensor):
            Cartesian coordinates comprising a point cloud, in XYZ order.

        resolution (float):
            The resolution at which to construct the 2D projection

        image_size (optional, 2-tuple of ints):
            By default, the size of the resultant projection is determined based on ``resolution`` and the
            range of x-y coordinates in ``coords``. An optional image size can be provided to override
            this behavior.

        padding_mode (str):
            Right now, only ``constant`` is supported.

        fill_value (str):
            Padding fill value for ``constant`` padding and for non-boundary pixels that did not have an assigned point.
            Default ``-1``.

        projection_map (tuple of tensor, tensor, (int, int)):
            An output of :func:`combustion.points.projection_mapping` to avoid mapping recomputation. Overrides
            ``resolution`` and ``image_size`` if given.

    Shapes
        * ``coords`` - :math:`(N, 3)` or :math:`(1, N, 3)`
        * Output - :math:`(H, W)`

    """
    # apply batch dim if not present
    coords = coords.view(coords.shape[-2], coords.shape[-1])

    if projection_map is not None:
        mapping, mask, (height, width) = projection_map
    else:
        mapping, mask, (height, width) = projection_mapping(coords, resolution, image_size)

    h_coords = mapping[..., 0]
    w_coords = mapping[..., 1]
    z = coords[mask, -1]

    # convert x,y coords into a single flat index for scatter_max
    flat_coords = h_coords.mul(width).add_(w_coords).long()

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

    projection = unoccluded_points.view(height, width)

    # apply padding
    # TODO: other padding modes would be nice, but are challenging to implement
    if padding_mode == "constant":
        if fill_value != -1:
            fill_no_data = torch.tensor(fill_value, device=unoccluded_points.device)
            projection[projection == -1] = fill_no_data
    elif padding_mode == "crop":
        raise NotImplementedError("replicate")
    elif padding_mode == "replicate":
        raise NotImplementedError("replicate")
    elif padding_mode == "reflect":
        raise NotImplementedError("reflect")
        # use reflection padding to create a slightly larger image
        padding = (width // 2, width // 2, height // 2, height // 2)
        padded = F.pad(projection.float().view(1, 1, height, width), padding, "reflect").squeeze_()

        # incorrect!
        # flip the larger image in various ways to fill in missing values
        _ = torch.where(padded == -1, padded.flip(dims=[-1]), padded)
        _ = torch.where(_ == -1, padded.flip(dims=[-2]), _)
        _ = torch.where(_ == -1, padded.flip(dims=[-2, -1]), _)
        projection = _[padding[1] : padding[1] + width, padding[0] : padding[0] + height].long()

        assert projection.shape == torch.Size([height, width])
    else:
        raise ValueError(f"Unknown padding_mode {padding_mode}")

    return projection
