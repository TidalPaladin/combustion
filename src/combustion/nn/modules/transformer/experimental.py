#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from abc import ABC, abstractmethod
from enum import IntEnum, Enum

from torch import Tensor
from typing import Any, Callable, Optional, Tuple, List, Type, Union
from math import sqrt
from functools import partial
from .common import MLP, DropPath, SqueezeExcite
from .point_transformer import KNNTail, KNNDownsample
from copy import deepcopy

class Unattention(nn.Module):

    def __init__(self, ratio: float, dim: int, num_heads: int = 1, **kwargs):
        super().__init__()
        assert 0 <= ratio <= 1
        self.ratio = ratio
        self.linear =  nn.Linear(dim, 1)
        self.mlp = MLP(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        #if self.training:
        #    return x
        L, N, D = x.shape
        attn = self.linear(x)


        #K = int(L * self.ratio)
        #attn = attn.softmax(dim=0)
        #idx = attn.topk(dim=0, k=K).indices
        #x[idx] = 0


        attn = attn.sigmoid().view(L, N)
        x[attn < self.ratio] = 0

        x = self.mlp(x)
        return x

class SoftmaxSE(SqueezeExcite):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.linear = nn.Linear(self.d_in, 1)

    def forward(self, x: Tensor) -> Tensor:
        L, N, D = x.shape
        L_dim = 0
        orig_x = x
        x = self.linear(x)
        x = x.softmax(dim=L_dim)
        x = (orig_x * x).mean(dim=L_dim)
        return orig_x + orig_x * self.se(x)
