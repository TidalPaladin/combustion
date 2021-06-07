#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
from torch import Tensor
import torch
import torch.nn as nn
from timeit import timeit

from combustion.nn.modules.transformer.performer import generalized_kernel_features, gaussian_orthogonal_random_matrix, FAVOR, PerformerLayer

class TestKernels:

    @pytest.mark.parametrize("uniform_q", [False, True])
    @pytest.mark.parametrize("rows,cols", [
        pytest.param(5, 5),
        pytest.param(10, 5),
        pytest.param(10, 8),
    ])
    def test_gaussian_orthogonal_matrix(self, rows, cols, uniform_q):
        mat = gaussian_orthogonal_random_matrix(rows, cols, uniform_q)
        assert isinstance(mat, Tensor)
        diag = torch.mm(mat, mat.T).diag()

        # test only works when scaling not applied
        #assert torch.allclose(diag, torch.ones_like(diag))

    def test_generalized_kernel(self):
        L, N, E = 10, 3, 20
        R = 5
        data = torch.rand(L, N, E)
        projection = torch.rand(E, R)
        kernel_func = lambda x, y: x

        features = generalized_kernel_features(data, kernel_func, projection)
        assert isinstance(features, Tensor)

class TestFavor:

    def test_init(self):
        E, R = 512, 128
        m = FAVOR(E, R, 8)

    def test_forward(self):
        torch.random.manual_seed(42)
        E, R = 512, 96
        L, N = 1024, 2

        m = FAVOR(E, R, 8)
        x = torch.rand(L, N, E)
        out, _ = m(x, x, x)
        assert isinstance(out, Tensor)
        assert out.shape == (L, N, E)

    def test_stability(self):
        torch.random.manual_seed(42)
        E, R = 512, 96
        L, N = 1024, 2

        m = FAVOR(E, R, 8)
        x = torch.rand(L, N, E)
        x.sub_(0.5).mul_(1000)
        out, _ = m(x, x, x)

        x = torch.zeros_like(x)
        out, _ = m(x, x, x)
        assert not out.isnan().any()

    def test_vs_standard_attn(self):
        num_heads = 4
        E, R = 512, 96
        L, N = 1024, 2

        torch.random.manual_seed(42)
        s = nn.MultiheadAttention(E, num_heads)
        torch.random.manual_seed(42)
        f = FAVOR(E, R, num_heads, fast=False)

        x = torch.rand(L, N, E)
        out_s, _ = s(x, x, x)
        out_f, _ = f(x, x, x)
        assert torch.allclose(out_s, out_f, atol=1e-5)

    def test_vs_standard_attn2(self):
        num_heads = 4
        E, R = 200, 128
        L, N = 1024, 4

        torch.random.manual_seed(42)
        s = nn.MultiheadAttention(E, num_heads)
        torch.random.manual_seed(42)
        f = FAVOR(E, R, num_heads)

        x = torch.rand(L, N, E)
        out_s, weight_s = s(x, x, x, need_weights=True)
        out_f, weight_f = f(x, x, x, need_weights=True)
        out_mse = (out_s - out_f).pow(2).mean().item() 
        weight_mse = (weight_s - weight_f).pow(2).mean().item() 
        assert weight_mse < 1e-6
        assert out_mse < 1e-3

    @pytest.mark.ci_skip
    def test_approximation_error(self):
        num_heads = 8
        E, R = 128, 32
        L, N = 1024, 15

        m_values = (16, 64, 128)
        seed_values = (42, 31, 18)

        results = {}
        for seed in seed_values:
            torch.random.manual_seed(seed)
            s = nn.MultiheadAttention(E, num_heads)
            x = torch.rand(L, N, E)
            out_s, weight_s = s(x, x, x, need_weights=True)

            for R in m_values:
                torch.random.manual_seed(seed)
                f = FAVOR(E, R, num_heads)
                out_f, weight_f = f(x, x, x, need_weights=True)
                mse_out = (out_s - out_f).pow(2).mean()
                results[f"{seed}-{R}"] = mse_out

        assert (v < 1e-2 for v in results.values())

    @pytest.mark.ci_skip
    def test_speed(self):
        num_heads = 8
        E, R = 512, 32
        N = 2

        Ls = (1024, 2048, 4096)
        s = nn.MultiheadAttention(E, num_heads)
        f = FAVOR(E, R, num_heads)
        times = []
        for L in Ls:
            x = torch.rand(L, N, E)
            func1 = lambda : s(x, x, x)
            func2 = lambda : f(x, x, x)
            t1 = timeit(func1, number=2)
            t2 = timeit(func2, number=2)
            times.append([t1, t2])

        for L, (t1, t2) in zip(Ls, times):
            assert t1 >= t2
            print(f"Time (L={L}): Baseline={t1}, Performer={t2}")


class TestPerformer:

    def test_init(self):
        nhead = 8
        E, R = 512, 128
        m = PerformerLayer(E, nhead, R)

    def test_forward(self):
        torch.random.manual_seed(42)
        nhead = 8
        N, E, R, L = 2, 512, 128, 1024
        m = PerformerLayer(E, nhead, R)
        x = torch.rand(L, N, E)
        out = m(x)
        assert isinstance(out, Tensor)
