#!/usr/bin/env python
# -*- coding: utf-8 -*-


import math
from typing import Tuple

import torch
from torch import Tensor


def cartesian_to_polar(x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
    r"""Converts Cartesian coordinates to polar coordinates.

    Args:
        x (:class:`torch.Tensor`):
            Input x coordinate

        y (:class:`torch.Tensor`):
            Input y coordinate

    Returns:
        Tuple of :math:`r`, :math:`\theta` polar coodinates
    """
    x = x.float()
    y = y.float()
    r = x.pow(2).add(y.pow(2)).sqrt()
    theta = torch.where(x != 0, y.div(x).arctan(), torch.zeros_like(x))
    # TODO why does theta.new_tensor(math.pi) fail in torchscript?
    theta = torch.where(x > 0, theta, theta + torch.tensor(math.pi, device=theta.device).type_as(theta))
    return r, theta


def polar_to_cartesian(r: Tensor, theta: Tensor) -> Tuple[Tensor, Tensor]:
    r"""Converts polar coordinates to Cartesian coordinates.

    Args:
        r (:class:`torch.Tensor`):
            Input radius

        theta (:class:`torch.Tensor`):
            Input angle

    Returns:
        Tuple of :math:`x`, :math:`y` Cartesian coodinates
    """
    x = r * theta.cos()
    y = r * theta.sin()
    return x, y
