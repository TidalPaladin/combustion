#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
import pytest
from torch import Tensor
import torch
import torch.nn as nn
from timeit import timeit

from combustion.nn.modules.transformer.performer import FAVOR, PerformerLayer, SoftmaxORF, SoftmaxHyp


class TestSoftmaxORF:

    def test_gaussian_orthogonal_matrix(self):
        L, N, E = 100, 2, 10
        torch.random.manual_seed(42)
        k = SoftmaxORF(E, E)
        query = torch.normal(0, 1, (L, N, E))
        key = torch.normal(0, 1, (L, N, E))
        query_sm = k(query, is_query=True)
        key_sm = k(key, is_query=False)

        out = torch.einsum("lne,pne->nlp", query, key).div_(math.sqrt(E)).softmax(dim=-1)
        out2 = torch.einsum("lne,pne->nlp", query_sm, key_sm)
        assert False

        # test only works when scaling not applied
        #assert torch.allclose(diag, torch.ones_like(diag))


class TestFavor:

    def test_init(self):
        E, nhead = 512, 8
        m = FAVOR(E, nhead)

    def test_forward(self):
        torch.random.manual_seed(42)
        E, nhead = 512, 8
        L, N = 1024, 2

        m = FAVOR(E, nhead)
        x = torch.rand(L, N, E)
        out, _ = m(x, x, x)
        assert isinstance(out, Tensor)
        assert out.shape == (L, N, E)

    def test_stability(self):
        torch.random.manual_seed(11)
        E, nhead = 512, 8
        L, N = 1024, 2

        m = FAVOR(E, nhead)
        x = torch.rand(L, N, E)
        x.sub_(0.5).mul_(1000)
        out, _ = m(x, x, x)

        x = torch.zeros_like(x)
        out, _ = m(x, x, x)
        assert not out.isnan().any()

    @pytest.mark.cuda_or_skip
    @pytest.mark.parametrize("num_heads", [1, 4, 8])
    @pytest.mark.parametrize("E", [32, 128, 256, 512])
    def test_vs_standard_attn2(self, E, num_heads, capsys):
        L, N = 1024, 2

        torch.random.manual_seed(11)
        s = nn.MultiheadAttention(E, num_heads)
        for p in s.parameters():
            torch.random.manual_seed(11)
            torch.nn.init.normal_(p, std=1)
        torch.random.manual_seed(42)
        f = FAVOR(E, num_heads, stabilizer=1e-16)
        for p in f.parameters():
            torch.random.manual_seed(11)
            torch.nn.init.normal_(p, std=1)
        s = s.cuda()
        f = f.cuda()

        f.eval()
        s.eval()

        x = torch.normal(0, 1, (L, N, E)).cuda()
        out_s, weight_s = s(x, x, x, need_weights=True)
        out_f, weight_f = f(x, x, x, need_weights=True)
        out_mse = (out_s - out_f).pow(2).mean().item() / out_s.mean().item()
        weight_mse = (weight_s - weight_f).pow(2).mean().item() / weight_s.mean().item()
        #assert weight_mse < 1e-6
        #assert out_mse < 1e-3
        with capsys.disabled():
            print(f"E={E}, nhead={num_heads}: Weight/Output = {weight_mse:.3E}/{out_mse:.3E}")

    @pytest.mark.skip
    @pytest.mark.ci_skip
    def test_approximation_error(self):
        num_heads = 8
        E, R = 128, 32
        L, N = 10, 128

        m_values = (16, 64, 128)
        seed_values = (42, 31, 18)

        results = {}
        for seed in seed_values:
            torch.random.manual_seed(seed)
            s = nn.MultiheadAttention(E, num_heads)
            x = torch.randn(L, N, E)
            out_s, weight_s = s(x, x, x, need_weights=True)

            for R in m_values:
                torch.random.manual_seed(seed)
                f = FAVOR(E, R, num_heads)
                out_f, weight_f = f(x, x, x, need_weights=True)
                mse_out = (out_s - out_f).pow(2).mean()
                results[f"{seed}-{R}"] = mse_out

        assert (v < 1e-2 for v in results.values())
        assert False

    @pytest.mark.skip
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


@pytest.mark.skip
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
