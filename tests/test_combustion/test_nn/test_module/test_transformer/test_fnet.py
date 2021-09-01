#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import torch

from combustion.nn.modules.transformer.fnet import FNet, FourierMixer, FourierTransformer


class TestFourierMixer:
    @pytest.mark.parametrize("nhead", [1, 2])
    @pytest.mark.parametrize("dtype", [torch.float, torch.half])
    def test_forward(self, nhead, dtype):
        L, N, D = 512, 2, 64
        x = torch.randn(L, N, D, dtype=dtype)
        l = FourierMixer(nhead)
        out = l(x)
        assert out.shape == (L, N, D)
        assert out.dtype == dtype


class TestFNetLayer:
    @pytest.mark.parametrize("nhead", [1, 2])
    def test_forward(self, nhead):
        L, N, D = 512, 2, 64
        x = torch.randn(L, N, D)
        l = FNet(D, 2 * D, nhead)
        out = l(x)
        assert out.shape == (L, N, D)


class TestFourierTransformer:
    @pytest.mark.parametrize("nhead", [1, 2])
    def test_forward(self, nhead):
        L, N, D = 512, 2, 64
        x = torch.randn(L, N, D)
        l = FourierTransformer(D, 2 * D, nhead)
        out = l(x)
        assert out.shape == (L, N, D)
