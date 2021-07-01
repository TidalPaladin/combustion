#!/usr/bin/env python
# -*- coding: utf-8 -*-


import math
import pytest
from torch import Tensor
import torch
import torch.nn as nn
from timeit import timeit

from combustion.nn.modules.transformer.fourier_features import RandomFeatures, OrthogonalFeatures

class TestRandomFeatures:

    @pytest.mark.parametrize("trainable", [False, True])
    @pytest.mark.parametrize("D,R", [
        pytest.param(128, 64),
        pytest.param(64, 64),
    ])
    def test_init(self, D, R, trainable):
        l = RandomFeatures(D, R, trainable=trainable)

    @pytest.mark.parametrize("D,R", [
        pytest.param(128, 64),
        pytest.param(64, 64),
    ])
    def test_forward(self, D, R):
        N, L = 2, 256
        l = RandomFeatures(D, R)
        x = torch.rand(N, L, D)
        out = l(x)
        assert out.shape == (N, 1, L, R)


class TestOrthogonalFeatures:

    @pytest.mark.parametrize("trainable", [False, True])
    @pytest.mark.parametrize("D,R", [
        pytest.param(128, 64),
        pytest.param(64, 64),
    ])
    def test_init(self, D, R, trainable):
        l = OrthogonalFeatures(D, R, trainable=trainable)

    @pytest.mark.parametrize("D,R,nhead", [
        pytest.param(128, 64, 1),
        pytest.param(64, 64, 1),
        pytest.param(128, 64, 4),
    ])
    def test_forward(self, D, R, nhead):
        N, L = 2, 256
        l = OrthogonalFeatures(D, R, num_heads=nhead)
        x = torch.rand(N, L, D)
        out = l(x)
        assert out.shape == (N, nhead, L, R)
