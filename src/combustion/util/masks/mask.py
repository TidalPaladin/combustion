#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List, Optional, Tuple

import torch
from torch import Tensor

from combustion.testing import is_scriptable


@is_scriptable
def index_assign_mask(mask: Tensor, indices: Tensor, values: Tensor) -> Tensor:
    r"""TorchScript compatable method to assign values to a mask tensor using indices in
    an adjacency list from :func:`get_adjacency`. When TorchScripting, only works
    for batched masks with dimensionality of 3 or lower.

    Args:
        mask:
            Mask to index in to

        indices:
            Indices at which to index

        values:
            Values to assign to ``mask`` at the indices in ``indices``

    Returns:
        Effectively, ``mask[indices] = values``

    Shapes:
        * ``mask`` - :math:`(*)`
        * ``indices`` - :math:`(N, *, C)` where :math:`C` == ``mask.ndim``
        * ``values`` - :math:`(N, *)`

    """
    ndim = indices.shape[-1]
    idx = list(indices.split((1,) * ndim, dim=-1))

    if not torch.jit.is_scripting():
        mask[idx] = values
        return mask

    # TorchScript won't let us index using a list since length is not statically
    # inferrable, so provide a few manual options for common dimensions
    # TODO: this should be fixable when tensor.nonzero(as_tuple=True) is scriptable
    elif ndim == 1:
        mask[idx[0]] = values
    elif ndim == 2:
        mask[idx[0], idx[1]] = values
    elif ndim == 3:
        mask[idx[0], idx[1], idx[2]] = values
    elif ndim == 4:
        mask[idx[0], idx[1], idx[2], idx[3]] = values
    elif ndim == 5:
        mask[idx[0], idx[1], idx[2], idx[3], idx[4]] = values
    else:
        raise RuntimeError(f"index_mask is not scriptable when indices.ndim > 4")
    return mask


@is_scriptable
def index_mask(mask: Tensor, indices: Tensor) -> Tensor:
    r"""TorchScript compatable method to index a mask tensor using an adjacency
    list from :func:`get_adjacency`. When TorchScripting, indexing can only be done
    for batched masks with dimensionality of 3 or lower.

    Args:
        mask:
            Mask to index in to

        indices:
            Indices at which to index

    Shapes:
        * ``mask`` - :math:`(*)`
        * ``indices`` - :math:`(*, C)` where :math:`C` == ``mask.ndim``
    """
    ndim = indices.shape[-1]
    idx = list(indices.split((1,) * ndim, dim=-1))

    if not torch.jit.is_scripting():
        return mask[idx]

    # TorchScript won't let us index using a list since length is not statically
    # inferrable, so provide a few manual options for common dimensions
    # TODO: this should be fixable when tensor.nonzero(as_tuple=True) is scriptable
    elif ndim == 1:
        return mask[idx[0]]
    elif ndim == 2:
        return mask[idx[0], idx[1]]
    elif ndim == 3:
        return mask[idx[0], idx[1], idx[2]]
    elif ndim == 4:
        return mask[idx[0], idx[1], idx[2], idx[3]]
    elif ndim == 5:
        return mask[idx[0], idx[1], idx[2], idx[3], idx[4]]
    else:
        raise RuntimeError(f"index_mask is not scriptable when indices.ndim > 4")


def _meshgrid(tensors: List[Tensor]) -> List[Tensor]:
    if not torch.jit.is_scripting():
        result = torch.meshgrid(*tensors)

    if len(tensors) == 2:
        result = torch.meshgrid(tensors[0], tensors[1])
    elif len(tensors) == 3:
        result = torch.meshgrid(tensors[0], tensors[1], tensors[2])
    elif len(tensors) == 4:
        result = torch.meshgrid(tensors[0], tensors[1], tensors[2], tensors[3])
    else:
        raise ValueError(f"Invalid number of tensors {len(tensors)} when scripting")

    return list(result)


@is_scriptable
def get_adjacency(
    nodes: Tensor, dims: List[int], diagonal: bool = True, self_loops: bool = False, dist: int = 1
) -> Tensor:
    r"""Given a set of node coordinates, return an adjacency list indicating
    all coordinats adjacent to each input node

    Args:
        nodes:
            Input coordinates

        dims:
            Indices in ``nodes`` for which to compute adjacency

        diagonal:
            Include diagonal neighbors in the adjacency list

        self_loops:
            Include self-loops in the adjacency list

        dist:
            Adjacency radius

    Shapes:
        * ``nodes`` - :math:`(N, C)` for :math:`N` coordinates of length-:math:`C`
        * Output - :math:`(N, A, C)` where :math:`A` is the number of adjacent nodes
    """
    # get locations of each positive label
    splits = (1,) * nodes.shape[-1]
    node_idx = torch.split(nodes, splits, dim=-1)

    # get all possible deltas
    # NOTE: torch.cartesian_prod not scriptable with variable dims; use two combinations calls instead
    diff = torch.arange(-dist, dist + 1).type_as(nodes)
    delta = torch.cat(
        (
            torch.combinations(diff, r=len(dims), with_replacement=True),
            torch.combinations(diff, r=len(dims), with_replacement=False).roll(1, dims=-1),
        ),
        dim=-2,
    )

    # filter deltas based on diagonal / self_loops
    is_self_loop = (delta == 0).all(dim=-1)
    is_diagonal = (delta != 0).sum(dim=-1) > 1
    keep = (~is_diagonal | torch.tensor(diagonal).type_as(is_diagonal)) & (
        ~is_self_loop | torch.tensor(self_loops).type_as(is_self_loop)
    )
    delta = delta[keep]
    D = delta.shape[-2]
    delta = torch.cat((delta.new_zeros(D, 1), delta), dim=-1)

    # for each of N coordinates, apply A unique delta values to get the N x A adjacency list
    N = nodes.shape[0]
    C = nodes.shape[-1]  # number of values in a coordinate
    A = delta.shape[-2]
    adjacency = (nodes.view(-1, 1, C) + delta.view(1, -1, C)).view(N, A, C)

    return adjacency


@is_scriptable
def clip_oob(
    adjacency: Tensor, upper_bound: List[int], lower_bound: Optional[List[int]] = None
) -> Tuple[Tensor, Tensor]:
    r"""Clips an adjacency list (e.g. output of :func:`get_adjacency`) so that all
    indices lie within the size range given by ``size``. Since :func:`get_adjacency`
    is unaware of the bounds, nodes that are at a border will have adjacencies that
    lie out of bounds.

    Let :math:`A` be the set of adjacent indicies for node :math:`N`. For all out of bounds
    adjacencies :math:`a_i \in A` for node :math:`N`, :math:`a_i` is replaced with an in-bounds
    adjacency :math:`a_j \in A`. This operation will result in duplicate edges.

    Args:
        adjacency:
            Adjacency list to clip

        upper_bound:
            Upper bounds for each dimension. Indices are considered in-bounds if
            ``(adjacency < upper_bound).all(dim=-1)``

        lower_bound:
            Lower bounds for each dimension. Indices are considered in-bounds if
            ``(adjacency >= lower_bound).all(dim=-1)``

    Returns:
        Two-tuple of tensors. The first is the clipped adjacency list, the second is
        a mask for out of bounds elements.

    Shapes:
        * ``adjacency`` - :math:`(N, A, C)` for :math:`N` nodes with :math:`A`
          adjacent length-:math:`C` indices each
        * Output - Same as ``adjacency``
    """
    if lower_bound is not None and len(lower_bound) != len(upper_bound):
        raise ValueError(f"lower_bound and upper_bound must be equal length: " f"{lower_bound} vs {upper_bound}")

    upper_bound = torch.tensor(upper_bound).type_as(adjacency).sub_(1)
    if lower_bound is None:
        _lower_bound = torch.zeros_like(upper_bound)
    else:
        _lower_bound = torch.tensor(lower_bound).type_as(adjacency)

    # handle oob by pointing OOB adjacencies to an IB coordinate
    adjacency.shape[-3]
    adjacency.shape[-2]
    C = adjacency.shape[-1]
    start = C - upper_bound.shape[-1]

    in_bounds = adjacency[..., start:] >= _lower_bound
    in_bounds &= adjacency[..., start:] <= upper_bound
    in_bounds = in_bounds.all(dim=-1)
    oob = ~in_bounds

    # early return if everything is in-bounds
    _ = oob.nonzero()
    if not _.numel():
        return adjacency, oob

    # find one adjacent node that is in-bounds for each node
    replacement_idx = in_bounds.max(dim=-1).indices

    # get node number for indexing
    replacement_node = _.split((1,) * _.shape[-1], dim=-1)[0].view(-1)

    adjacency[oob] = adjacency[replacement_node, replacement_idx[replacement_node]]

    return adjacency, oob


@is_scriptable
def border_dist(mask: Tensor) -> Tensor:
    r"""Computes pixel-wise distances to the mask perimeter. This function
    generalizes to masks of arbitrary dimensions.

    Args:
        mask (:class:`torch.Tensor`):
            Input mask

    Shape:
        * ``mask`` - :math:`(B, *)`
        * Output - :math:`(B, C, *)` where :math:`C` is the number of dimensions in :math:`*`

    Example:
        >>> mask = torch.tensor([[
        >>>     [0, 0, 1, 0, 0],
        >>>     [0, 1, 1, 1, 0],
        >>>     [1, 1, 1, 1, 1],
        >>>     [0, 1, 1, 1, 0],
        >>>     [0, 0, 1, 0, 0],
        >>> ]])
        >>> dist = edge_dist(mask)
    """
    mask.shape[0]
    ndim = mask.ndim
    dims = mask.shape[1:]
    mask = mask > 0

    # build a grid of distances from endpoints for each dimension
    tensors: List[Tensor] = []
    flipped_tensors: List[Tensor] = []
    for i in range(-1, -ndim, -1):
        t1 = torch.arange(mask.shape[i], device=mask.device)
        t2 = torch.arange(mask.shape[i] - 1, -1, -1, device=mask.device)
        for j in range(-1, -ndim, -1):
            if i != j:
                t1.unsqueeze_(j)
                t2.unsqueeze_(j)
        tensors.append(t1.expand_as(mask))
        flipped_tensors.append(t2.expand_as(mask))
    pos = torch.stack(tensors + flipped_tensors, dim=1)
    C = pos.shape[1]
    assert C == 2 * (ndim - 1)
    return pos


@is_scriptable
def edge_dist(mask: Tensor) -> Tensor:
    r"""Computes pixel-wise distances to the mask perimeter. This function
    generalizes to masks of arbitrary dimensions.

    Args:
        mask (:class:`torch.Tensor`):
            Input mask

    Shape:
        * ``mask`` - :math:`(B, *)`
        * Output - :math:`(B, C, *)` where :math:`C` is the number of dimensions in :math:`*`

    Example:
        >>> mask = torch.tensor([[
        >>>     [0, 0, 1, 0, 0],
        >>>     [0, 1, 1, 1, 0],
        >>>     [1, 1, 1, 1, 1],
        >>>     [0, 1, 1, 1, 0],
        >>>     [0, 0, 1, 0, 0],
        >>> ]])
        >>> dist = edge_dist(mask)
    """
    mask.shape[0]
    ndim = mask.ndim
    dims = mask.shape[1:]
    mask = mask > 0

    # build a grid of distances from endpoints for each dimension
    pos = border_dist(mask)
    C = pos.shape[1]
    assert C == 2 * (ndim - 1)

    # build array of reset values
    mask.unsqueeze_(1)
    resets = pos + 1
    resets[mask.expand_as(resets)] = 0

    # build a grid deltas that will reset running distances on non-mask pixels
    tensors: List[Tensor] = []
    for i in range(C):
        dim = -(i % (ndim - 1)) - 1
        if i < C / 2:
            t = torch.cummax(resets[:, i, ...], dim=dim).values
        else:
            t = resets[:, i, ...].flip(dims=(dim,)).cummax(dim=dim).values.flip(dims=(dim,))
        tensors.append(t)
    reset_val = torch.stack(tensors, dim=1)

    # apply deltas to distance grid and set non-mask pixels to -1
    result = pos - reset_val
    result[(~mask).expand_as(result)] = -1
    return result


@is_scriptable
def inverse_edge_dist(dist: Tensor, mask: Optional[Tensor] = None, weights: Optional[Tensor] = None) -> Tensor:
    r"""Performs the inverse of :func:`edge_dist`, generating a heatmap that counts the total
    number of locations pointing to a given coordinate.

    Args:
        dist:
            Distance offsets, such as those generated by :func:`edge_dist`

        mask:
            Optional mask tensor used to indicated valid positions that should contribute
            to the resultant heatmap.

        weights:

    Shape:
        * ``dist`` - :math:`(B, C, *)` where :math:`C` is the number of dimensions in :math:`*`
        * ``mask`` - :math:`(B, *)`
        * Output - :math:`(B, *)`
    """
    B = dist.shape[0]
    C = dist.shape[1]
    dims = dist.shape[2:]
    ndim = dist.ndim - 2

    # validate weight input if given
    if weights is None:
        _weights = torch.ones_like(dist).float()
    elif weights.shape != dist.shape:
        raise ValueError("`weights` should be the same shape as `dist`: " f"{weights.shape} vs {dist.shape}")
    else:
        _weights = weights.float()

    # base values are the coordinates at each location in the tensor
    ranges = [torch.arange(x, device=dist.device) for x in dims]
    base = torch.stack(_meshgrid(ranges))

    # shape matching
    # unsqeeze batch and C dims from base
    # unsqeeze base coordinate dim from offset
    base = base.unsqueeze_(0).unsqueeze_(2)
    dist = dist.unsqueeze_(1)
    base, offset = torch.broadcast_tensors(base, dist)
    dist.squeeze_(1)
    assert base.shape[0] == offset.shape[0] == B
    assert base.shape[1] == offset.shape[1] == ndim
    assert base.shape[2] == offset.shape[2] == C

    # offset values are the aboslute displacements given in dist
    # all input values are positive, so flip sign of backward-pointing displacements
    offset = offset.clone()
    offset.clamp_min_(0)
    offset[:, :, :2, ...].neg_()

    # zero out displacments that aren't associated with a base coordinate
    for i in range(ndim):
        idx = [x for x in range(i, C, ndim)]
        offset[:, i, idx, :, :] = 0

    # apply offset to base coordinates and clamp to lie within grid
    points = base + offset
    for i in range(ndim):
        points[:, i, ...].clamp_(min=0, max=dims[i] - 1)
    points = points.view(B, 2, -1).swapdims(-2, -1)

    # build final coordinate list that includes batch index
    N = points.shape[-2]
    batch_idx = torch.arange(B, device=points.device).view(B, 1, 1).expand(B, N, 1)
    points = torch.cat((batch_idx, points), dim=-1).view(-1, 3)
    _weights = _weights.flatten()

    # if a mask was given, discard distances emitted from mask=False locations
    if mask is not None:
        mask = (mask > 0).unsqueeze_(1).expand_as(base[:, 0, :, ...])
        nonmask_idx = mask.contiguous().view(-1)
        points = points[nonmask_idx, :]
        _weights = _weights[nonmask_idx]
        N = points.shape[-2]

    # get list of unique coordinates and their counts
    unique_idx, inverse, counts = torch.unique(points, sorted=False, return_counts=True, return_inverse=True, dim=-2)
    total_weight = torch.zeros_like(counts, dtype=torch.float).scatter_add(-1, inverse, _weights)
    total_weight = total_weight.view(-1, 1)

    # fill heatmap at coordinate locations with associated count
    heatmap = torch.zeros_like(dist[:, 0, ...], dtype=torch.float)
    index_assign_mask(heatmap, unique_idx, total_weight)

    return heatmap


@is_scriptable
def get_edges(mask: Tensor, diagonal: bool = True) -> Tensor:
    r"""Given a binary mask, extract coodinates indicating tensor indicating edges.

    Args:
        mask (:class:`torch.Tensor`):
            Binary mask to extract edges from

        diagonal (bool):
            Whether or not diagonals should be included in edge detection

    Shapes:
        * ``mask`` - :math:`(B, *)`
        * Output - :math:`(N, C)` where :math:`C` is ``mask.ndim``.
    """
    mask.shape[0]
    ndim = mask.ndim
    dims = mask.shape[1:]
    mask = mask > 0

    nodes = mask.nonzero()
    indexable_nodes = nodes.split((1,) * ndim, dim=-1)
    N = nodes.shape[0]

    if N == 0:
        return nodes

    adjacency = get_adjacency(nodes, list(range(1, ndim)), diagonal, self_loops=False)
    adjacency, oob = clip_oob(adjacency, mask.shape[1:])
    idx = adjacency.split((1,) * ndim, dim=-1)

    torch.full_like(mask, 0)
    is_edge = (~index_mask(mask, adjacency)).view(N, -1).any(dim=-1, keepdim=True)
    border_node = oob.any(dim=-1, keepdim=True)
    pos = is_edge.logical_or_(border_node)
    return nodes[pos.view(-1)]


@is_scriptable
def expand_mask(mask: Tensor, amount: int = 1, diagonal: bool = True) -> Tensor:
    r"""Expands a binary mask by a given amount

    Args:
        mask:
            The mask to expand

        amount:
            How much to expand by

        diagonal:
            If ``False``, only expand the mask along non-diagonals

    Shapes:
        * ``mask`` - :math:`(B, *)`
        * Output - Same as input
    """
    mask.shape[0]
    ndim = mask.ndim
    dims = mask.shape[1:]
    mask = mask > 0

    if amount == 0:
        return mask

    edges = get_edges(mask)
    adjacency = get_adjacency(edges, list(range(1, ndim)), diagonal, dist=amount)
    adjacency, oob = clip_oob(adjacency, mask.shape[1:])
    index_assign_mask(mask, adjacency, torch.tensor(True).type_as(mask))
    return mask


@is_scriptable
def contract_mask(mask: Tensor, amount: int = 1, diagonal: bool = False) -> Tensor:
    r"""Contracts a binary mask by a given amount

    Args:
        mask:
            The mask to contract

        amount:
            How much to contract by

        diagonal:
            If ``False``, only contract the mask along non-diagonals

    Shapes:
        * ``mask`` - :math:`(B, *)`
        * Output - Same as input
    """
    invmask = (mask > 0).logical_not_()
    invmask = expand_mask(invmask, amount, diagonal=False)
    return invmask.logical_not_()


@is_scriptable
def min_spacing(mask: Tensor) -> Tensor:
    mask = mask > 0
    B = mask.shape[0]
    ndim = mask.ndim

    # compute edge_dist on mask=False locations
    invmask = ~(mask > 0)
    dist = edge_dist(invmask)

    # per-location reduction of max distance from either end along that dimension
    tensors: List[Tensor] = []
    C = dist.shape[1]
    for i in range(ndim - 1):
        nextpos = i + C // 2
        t = torch.max(dist[:, i : i + 1, ...], dist[:, nextpos : nextpos + 1, ...])
        tensors.append(t)
    dist = torch.cat(tensors, dim=1)

    # fill non-edge locations with a big value so they aren't included in min dist calculation
    MAX_VAL = torch.tensor(mask.shape[1:]).type_as(dist).sum()
    edge_mask = torch.zeros_like(mask)
    edges = get_edges(invmask)
    index_assign_mask(edge_mask, edges, torch.tensor(True).type_as(edge_mask))
    edge_mask.unsqueeze_(1)
    idx = edge_mask.logical_not_().expand_as(dist).nonzero()
    index_assign_mask(dist, idx, MAX_VAL)

    return dist.view(B, -1).amin(dim=-1)


@is_scriptable
def connect_masks(mask: Tensor, spacing: int = 1) -> Tensor:
    r"""Attempts to uniformly expand a mask with multiple contiguous regions so that the
    regions are separated by a minimum spacing. The minimum spacing between contiguous
    regions (or mask border) is computed using :func:`min_spacing`, and the masks are expanded
    uniformly by that spacing value.

    .. warning::
        This method assumes multiple contiguous regions in the input because detecting unique
        instances is slow. For inputs with only one contiguous region, the mask will be expanded
        based on distance from the borders of the mask.

    Args:
        mask (:class:`torch.Tensor`):
            Binary mask to connect

        spacing:
            Minimum distance between masks after expansion

    Shapes:
        * ``mask`` - :math:`(H, W)`
        * Output - :math:`(H, W)`
    """
    B = mask.shape[0]

    # compute min spacing between masks
    expand_amount = min_spacing(mask).sub_(spacing - 1).clamp_min_(0).view(B, 1)

    # expand each mask by min_spacing value
    tensors: List[Tensor] = []
    for m, a in zip(mask, expand_amount):
        m.unsqueeze_(0)
        m = expand_mask(m, a.item(), diagonal=False)
        m.squeeze_(0)
        tensors.append(m)
    return torch.stack(tensors, dim=0)


@is_scriptable
def get_instances(mask: Tensor) -> Tensor:
    r"""Given a binary mask, extract a new mask indicating contiguous
    instances. The resultant mask will use ``0`` to indicate background
    classes, and will begin numbering instance regions with ``1``.

    This method identifies connected components via nearest-neighbor message passing.
    The runtime is a function of the diameter of the largest connected component.

    .. note::
        When TorchScripting, the global RNG must be seeded in order to produce deterministic
        instance labels. When not TorchScripting, RNG seed is set automtically to ensure
        determinstic output.

    Args:
        mask (:class:`torch.Tensor`):
            Binary mask to extract instances

    Shapes:
        * ``mask`` - :math:`(H, W)`
        * Output - :math:`(H, W)`
    """
    H, W = mask.shape[-2:]
    mask = mask > 0

    if not mask.any():
        return mask

    # assign each positive location a unique instance label
    _ = torch.arange(0, H * W, device=mask.device)

    if torch.jit.is_scripting():
        generator = None
    else:
        generator = torch.Generator(device=mask.device)
        generator.manual_seed(42)

    instances = (
        torch.multinomial(
            torch.ones(H * W, dtype=torch.float, device=mask.device), H * W, replacement=False, generator=generator
        )
        .view(H, W)
        .long()
    )
    instances[~mask] = 0

    # get locations of each positive label
    mask.unsqueeze_(0)
    nodes = mask.nonzero()
    mask.squeeze_(0)
    N = nodes.shape[0]
    node_idx = torch.split(nodes, [1, 1, 1], dim=-1)[1:]

    # get adjacency list for each positive node
    adjacency = get_adjacency(nodes, dims=(-1, -2), diagonal=True, self_loops=True)[..., 1:]
    adjacency, _ = clip_oob(adjacency, mask.shape[-2:])
    adjacency_idx = torch.split(adjacency, [1, 1], dim=-1)
    A = adjacency.shape[-2]

    passed_messages = instances[adjacency_idx].view(N, A)

    # iteratively pass instance label to neighbors
    # adopt instance label of max(self, neighbors)
    # NOTE: try to buffer things and operate in place where possible for speed
    old_instances = instances
    new_instances = instances.clone()
    adjacency_buffer = adjacency.new_empty(N)
    adjacency.new_empty(N)
    while True:
        # passed_messages = old_instances[adjacency_idx].view(N, A)
        passed_messages = index_mask(old_instances, adjacency).view(N, A)
        torch.amax(passed_messages, dim=-1, out=adjacency_buffer)
        index_assign_mask(new_instances, nodes[..., 1:], adjacency_buffer.view(-1, 1))

        # if nothing was updated, we're done
        diff = new_instances[mask] != old_instances[mask]
        if not diff.any():
            break

        _ = new_instances
        new_instances = old_instances
        old_instances = _

    # convert unique instance labels into a 1-indexed set of consecutive labels
    unique_instances = torch.unique(instances)
    for new_ins, old_ins in enumerate(unique_instances):
        if old_ins == 0:
            continue
        instances[instances == old_ins] = new_ins

    return instances.long()
