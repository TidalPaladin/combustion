#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import torch
from combustion.nn.modules.transformer.fnet import FNet, FourierDownsample, FourierUpsample, FourierMixer, FourierTransformer


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

class TestFourierDownsample:

    @pytest.mark.parametrize("nhead", [1, 2])
    @pytest.mark.parametrize("dtype", [torch.float, torch.half])
    def test_forward(self, nhead, dtype):
        L, N, D = 512, 2, 64
        x = torch.randn(L, N, D, dtype=dtype)
        l = FourierDownsample(nhead)
        out = l(x)
        assert out.shape == (L // 2 + 1, N, D)
        assert out.dtype == dtype

class TestFourierUpsample:

    @pytest.mark.parametrize("nhead", [1, 2])
    @pytest.mark.parametrize("dtype", [torch.float, torch.half])
    def test_forward(self, nhead, dtype):
        L, N, D = 512, 2, 64
        x = torch.randn(L // 2 + 1, N, D, dtype=dtype)
        l = FourierUpsample(nhead)
        out = l(x)
        assert out.shape == (L, N, D)
        assert out.dtype == dtype

class TestFNetLayer:

    @pytest.mark.parametrize("nhead", [1, 2])
    def test_forward(self, nhead):
        L, N, D = 512, 2, 64
        x = torch.randn(L, N, D)
        l = FNet(D, 2*D, nhead)
        out = l(x)
        assert out.shape == (L, N, D)
        assert False

    @pytest.mark.parametrize("nhead", [1, 2])
    def test_down_up(self, nhead):
        L, N, D = 512, 2, 64
        x = torch.randn(L, N, D)
        l0 = FNet(D, D, nhead)
        l1 = FNet(D, 2*D, nhead, dout=2*D)
        l2 = FourierDownsample(nhead)
        l3 = FNet(2*D, 2*D, nhead)
        l4 = FourierUpsample(nhead)
        l5 = FNet(2*D, D, nhead, dout=D)

        t0 = l0(x)
        t1 = l1(t0)
        down = l2(t1)
        down = l3(down)
        up = l4(down)
        final = l5(up)
        out = t0 + final

        assert out.shape == (L, N, D)

class TestFourierTransformer:

    @pytest.mark.parametrize("nhead", [1, 2])
    def test_forward(self, nhead):
        L, N, D = 512, 2, 64
        x = torch.randn(L, N, D)
        l = FourierTransformer(D, 2*D, nhead)
        out = l(x)
        assert out.shape == (L, N, D)
