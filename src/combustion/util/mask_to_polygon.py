#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List

import torch
import torch.nn.functional as F
from torch import Tensor


@torch.jit.script
def _get_edge_kernel() -> Tensor:
    d_kernel = torch.tensor(
        [
            [1, 0],
            [0, -1],
        ],
        dtype=torch.float,
    ).view(1, 1, 2, 2)

    w_kernel = (
        torch.tensor(
            [
                [1, -1],
                [0, 0],
            ]
        )
        .float()
        .view(1, 1, 2, 2)
    )

    h_kernel = torch.tensor(
        [
            [1, 0],
            [-1, 0],
        ]
    ).view(1, 1, 2, 2)

    return torch.cat([d_kernel, w_kernel, h_kernel], dim=0)


@torch.jit.script
def mask_to_edges(mask: Tensor) -> Tensor:
    r"""Given a binary mask, extract a binary mask indicating edges.

    Args:
        mask (:class:`torch.Tensor`):
            Binary mask to extract edges from

    Shapes:
        * ``mask`` - :math:`(H, W)`
        * Output - :math:`(H, W)`
    """
    H, W = mask.shape[-2:]

    mask = (mask > 0).float().view(1, 1, H, W)
    pad_mask = F.pad(mask, (1, 1, 1, 1), mode="constant", value=0.0)
    with torch.no_grad():
        kernel = _get_edge_kernel().type_as(pad_mask)
        diff = F.conv2d(pad_mask, kernel).abs_().sum(dim=1)
        diff_lt = diff[..., 1:, 1:] * mask
        diff_rb = diff[..., :-1, :-1] * mask
        edge_mask = (diff_lt + diff_rb) > 0

    return edge_mask.view(H, W)


@torch.jit.script
def mask_to_instances(mask: Tensor) -> Tensor:
    r"""Given a binary mask, extract a new mask indicating contiguous
    instances. The resultant mask will use ``0`` to indicate background
    classes, and will begin numbering instance regions with ``1``.

    .. note::
        This method relies on an iterative loop of max pooling operations,
        and thus may be slow for some inputs

    Args:
        mask (:class:`torch.Tensor`):
            Binary mask to extract instances

    Shapes:
        * ``mask`` - :math:`(H, W)`
        * Output - :math:`(H, W)`
    """
    H, W = mask.shape[-2:]

    # naive way
    mask = mask > 0
    grid = torch.meshgrid(torch.arange(H), torch.arange(W))
    instances = (grid[0] * H + grid[1] + 1).float().to(mask.device)

    instances[~mask] = 0
    while True:
        _ = F.max_pool2d(instances.view(1, 1, H, W), 3, stride=1, padding=1).view(H, W)
        _[~mask] = 0
        if (_ == instances).all():
            break
        instances = _

    unique_instances = torch.unique(instances)
    for new_ins, old_ins in enumerate(unique_instances):
        if old_ins == 0:
            continue
        instances[instances == old_ins] = new_ins

    return instances.long()


@torch.jit.script
def mask_to_box(mask: Tensor) -> Tensor:
    r"""Given a binary mask, extract bounding boxes for each contiguous
    region.

    .. note::
        This method relies on :func:`combustion.util.mask_to_instances`,
        so the same speed remark applies.

    Args:
        mask (:class:`torch.Tensor`):
            Binary mask to extract anchor boxes from

    Returns:
        Bounding box tensors in :math:`x_1, y_1, x_2, y_2` order.
        The :math:`i`'th box corresponds to the :math:`i`'th contiguous instance.

    Shapes:
        * ``mask`` - :math:`(H, W)`
        * Output - :math:`(N, 4)`
    """
    mask = mask > 0
    edges = mask_to_edges(mask)
    instances = mask_to_instances(edges)
    unique_instances = torch.unique(instances)
    result = []
    for i in unique_instances:
        if i == 0:
            continue
        coords = (instances == i).nonzero()
        lower = coords.amin(dim=-2)
        upper = coords.amax(dim=-2)
        result.append(torch.cat((lower, upper), dim=-1))
    return torch.stack(result, dim=0)


@torch.jit.script
def mask_to_polygon(mask: Tensor) -> List[Tensor]:
    r"""Given a binary mask, extract bounding polygons for each contiguous
    region.

    .. note::
        This method relies on :func:`combustion.util.mask_to_instances`,
        so the same speed remark applies.

    Args:
        mask (:class:`torch.Tensor`):
            Binary mask to extract polygons from

    Returns:
        A list of polygon tensors in :math:`x_1, y_1` order. The :math:`i`'th
        polygon corresponds to the :math:`i`'th contiguous instance.

    Shapes:
        * ``mask`` - :math:`(H, W)`
        * Polygons - :math:`(N, 2)`
    """
    mask = mask > 0
    edges = mask_to_edges(mask)
    instances = mask_to_instances(edges)
    unique_instances = torch.unique(instances)
    result = []
    for i in unique_instances:
        if i == 0:
            continue
        coords = (instances == i).nonzero()
        result.append(coords)

    return result
