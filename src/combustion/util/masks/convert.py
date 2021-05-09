#!/usr/bin/env python
# -*- coding: utf-8 -*-


from typing import List, Tuple

import torch
from torch import Tensor

from .mask import get_edges, get_instances, index_assign_mask

@torch.jit.script
def mask_to_box(mask: Tensor) -> Tensor:
    r"""Given a binary mask, extract bounding boxes for each contiguous
    region.

    .. note::
        This method relies on :func:`combustion.util.get_instances`,
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
    mask = (mask > 0)
    instances = get_instances(mask)
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


def mask_to_polygon(mask: Tensor) -> List[Tensor]:
    r"""Given a binary mask, extract bounding polygons for each contiguous
    region.

    .. note::
        This method relies on :func:`combustion.util.get_instances`,
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
    mask.unsqueeze_(0)
    edges = get_edges(mask, diagonal=True)
    mask.fill_(torch.tensor(False, device=mask.device))
    index_assign_mask(mask, edges, torch.tensor(True, device=mask.device))
    mask.squeeze_(0)
    instances = get_instances(mask)
    unique_instances = torch.unique(instances)
    result = []
    for i in unique_instances:
        if i == 0:
            continue
        coords = (instances == i).nonzero()
        result.append(coords)

    return result
