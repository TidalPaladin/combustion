#!/usr/bin/env python
# -*- coding: utf-8 -*-

from copy import deepcopy
from typing import Optional
from math import log, pi

import torch
import torch.nn as nn
from torch import Tensor
from .position import LearnableFourierFeatures

class AttentionConv2d(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride):
        super().__init__()
        self.query = nn.Parameter(torch.randn([1, 1, out_channels]))
        self.k_proj = nn.Linear(in_channels, out_channels)
        self.v_proj = nn.Linear(in_channels, out_channels)
        self.pos = LearnableFourierFeatures(2, out_channels, out_channels)

    def forward(self, x: Tensor) -> Tensor:
        N, D, H, W = x.shape
        x = x.view(N, D, -1).movedim(-1, 0)

        grid = LearnableFourierFeatures.from_grid(dims=(H, W), proto=x)
        pos = self.pos()
        K = self.k_proj(x)
        V = self.v_proj(x)
        import pdb; pdb.set_trace()
        ...

if __name__ == "__main__":
    l = AttentionConv2d(3, 32, 3, 1)
    x = torch.rand(1, 3, 32, 32)
    out = l(x)
