#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
from torch import Tensor
import pytest
from combustion.nn.modules.transformer.position import RelativePositionalEncoder, LearnableFourierFeatures

class TestRelativePositionalEncoder:

    def test_forward(self):
        B, N = 2, 100
        coords = torch.normal(0, 10.0, (B, N, 3))

        l = RelativePositionalEncoder(3, 32)
        assert False

class TestLearnableFourierFeatures:

    def test_from_grid(self):
        H, W = 10, 12
        coords = LearnableFourierFeatures.from_grid(dims=(H, W))
        assert coords.shape == (H*W, 1, 2)
        assert (coords[0, 0] == torch.tensor([0, 0])).all()
        assert (coords[1, 0] == torch.tensor([0, 1])).all()
        assert (coords[W, 0] == torch.tensor([1, 0])).all()

    @pytest.mark.parametrize("C", [1, 2, 3])
    def test_forward(self, C):
        L, N = 100, 2
        D = 24
        Nf = 32

        coords = torch.normal(0, 1, (L, N, C))
        l = LearnableFourierFeatures(C, Nf, D)
        pos = l(coords)
        assert pos.shape == (L, N, D)
