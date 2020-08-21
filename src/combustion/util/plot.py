#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from matplotlib import cm
from torch import Tensor


def apply_colormap(inputs: Tensor, cmap: str = "gnuplot") -> Tensor:
    r"""Applies a Matplotlib colormap to a tensor, returning a tensor.

    Args:
        inputs (:class:`torch.Tensor`):
            Tensor to apply colormap to

        cmap (str):
            The Matplotlib colormap

    Shape
        * ``inputs`` - :math:`(B, 1, *)`
        * Output - :math:`(B, 4, *)`
    """
    channel_dim = 1
    batch_dim = 0
    if inputs.shape[channel_dim] != 1:
        raise ValueError(f"Expected channel size = 1, but found inputs.shape == {inputs.shape}")
    cmap = cm.get_cmap(cmap)

    output_shape = list(inputs.shape)
    output_channels = 4
    output_shape[channel_dim] = output_channels

    output: List[Tensor] = []
    _ = inputs.flatten(start_dim=channel_dim, end_dim=-1).numpy()
    for batch_elem in _:
        mapped = torch.as_tensor(cmap(batch_elem), device=inputs.device, dtype=torch.float).transpose_(0, 1)
        output.append(mapped)

    return torch.stack(output, dim=batch_dim).view(*output_shape)
