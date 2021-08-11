#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
from timeit import timeit

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from combustion.nn.modules.transformer.performer import FAVOR, PerformerEncoderLayer, PointPerformer, SoftmaxORF


class TestSoftmaxORF:
    @pytest.mark.skip
    def test_gaussian_orthogonal_matrix(self):
        L, N, E = 100, 2, 10
        torch.random.manual_seed(42)
        k = SoftmaxORF(E, E)
        query = torch.normal(0, 1, (L, N, E))
        key = torch.normal(0, 1, (L, N, E))
        query_sm = k(query, is_query=True)
        key_sm = k(key, is_query=False)

        out = torch.einsum("lne,pne->nlp", query, key).div_(math.sqrt(E)).softmax(dim=-1)
        torch.einsum("lne,pne->nlp", query_sm, key_sm)
        assert False

        # test only works when scaling not applied
        # assert torch.allclose(diag, torch.ones_like(diag))


# class TestLogSoftmax:
#
#    def test_gaussian_orthogonal_matrix(self):
#        L, N, E = 100, 2, 10
#        torch.random.manual_seed(42)
#        k = LogSoftmax(E, 1)
#        query = torch.normal(0, 1, (L, N, 1, E))
#        key = torch.normal(0, 1, (L, N, 1, E))
#        query_sm = k(query, is_query=True).view(L, N, E)
#        key_sm = k(key, is_query=False).view(L, N, E)
#
#        out = torch.einsum("lne,pne->nlp", query.view(L, N, E), key.view(L, N, E)).div_(math.sqrt(E)).log_softmax(dim=-1)
#        out2 = torch.einsum("lne,pne->nlp", query_sm, key_sm)
#        assert False
#
#        # test only works when scaling not applied
#        #assert torch.allclose(diag, torch.ones_like(diag))


class TestSoftmaxSquare:
    def test_gaussian_orthogonal_matrix(self):
        L, N, E = 100, 2, 10
        torch.random.manual_seed(42)
        k = SoftmaxSquare(E, 1)
        query = torch.normal(0, 1, (L, N, 1, E))
        key = torch.normal(0, 1, (L, N, 1, E))
        query_sm = k(query, is_query=True).view(L, N, E)
        key_sm = k(key, is_query=False).view(L, N, E)

        out = (
            torch.einsum("lne,pne->nlp", query.view(L, N, E), key.view(L, N, E))
            .div_(math.sqrt(E))
            .softmax(dim=-1)
            .pow(2)
        )
        D_inv = torch.einsum("lne,pne->nlp", query_sm, key_sm.sum(dim=0, keepdim=True)).clamp_min(1e-6).reciprocal()
        D_inv.pow(2) * torch.einsum("lne,pne->nlp", query_sm, key_sm)
        assert False

        # test only works when scaling not applied
        # assert torch.allclose(diag, torch.ones_like(diag))


class TestFavor:
    def test_init(self):
        E, nhead = 512, 8
        FAVOR(E, nhead)

    def test_forward(self):
        torch.random.manual_seed(42)
        E, nhead = 512, 8
        L, N = 1024, 2

        m = FAVOR(E, nhead)
        x = torch.rand(L, N, E)
        out, _ = m(x, x, x)
        assert isinstance(out, Tensor)
        assert out.shape == (L, N, E)

        FAVOR.compute_model_gini(m)

    def test_forward_cross(self):
        torch.random.manual_seed(42)
        E1, nhead = 512, 8
        E2 = 1024
        L1, N = 1024, 2
        L2 = 512

        m = FAVOR(E1, nhead, kdim=E2, vdim=E2)
        x = torch.rand(L1, N, E1)
        y = torch.rand(L2, N, E2)
        out, _ = m(x, y, y)
        assert isinstance(out, Tensor)
        assert out.shape == (L1, N, E1)

        FAVOR.compute_model_gini(m)
        assert False

    @pytest.mark.parametrize("half", [False, True])
    def test_stability(self, half):
        torch.random.manual_seed(11)
        E, nhead = 512, 8
        L, N = 1024, 2

        m = FAVOR(
            E,
            nhead,
        ).cuda()
        x = torch.normal(0, 10, (L, N, E)).cuda()
        with torch.cuda.amp.autocast(enabled=half):
            out, _ = m(x, x, x)
            assert not out.isnan().any()

            # x = torch.zeros_like(x)
            # out, _ = m(x, x, x)
            # assert not out.isnan().any()

    @pytest.mark.parametrize("num_heads", [1, 4, 8])
    @pytest.mark.parametrize("E", [32, 128, 256, 512])
    def test_vs_standard_attn2(self, E, num_heads, capsys):
        L, N = 64, 2
        std = 1

        torch.random.manual_seed(11)
        s = nn.MultiheadAttention(E, num_heads)
        torch.random.manual_seed(42)
        f = FAVOR(E, num_heads, stabilizer=1e-5, trainable=False)

        f.eval()
        s.eval()

        x = torch.normal(0, std, (L, N, E))
        out_s, weight_s = s(x, x, x, need_weights=True)
        out_f, weight_f = f(x, x, x, need_weights=True)
        out_mse = (out_s - out_f).pow(2).mean().item() / out_s.mean().abs().item()
        weight_mse = (weight_s - weight_f).pow(2).mean().item() / weight_s.mean().abs().item()
        # assert weight_mse < 1e-6
        # assert out_mse < 1e-3
        maxdelta = (weight_s.max() - weight_f[weight_s == weight_s.max()].mean()).abs().item()
        mindelta = (weight_s.min() - weight_f[weight_s == weight_s.min()].mean()).abs().item()
        with capsys.disabled():
            print(
                f"E={E}, nhead={num_heads}, std={std}: Weight/Output/Max/Min = {weight_mse:.3E}/{out_mse:.3E}/{maxdelta:.3E}/{mindelta:.3E}"
            )

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

            def func1():
                return s(x, x, x)

            def func2():
                return f(x, x, x)

            t1 = timeit(func1, number=2)
            t2 = timeit(func2, number=2)
            times.append([t1, t2])

        for L, (t1, t2) in zip(Ls, times):
            assert t1 >= t2
            print(f"Time (L={L}): Baseline={t1}, Performer={t2}")


class TestPerformer:
    @pytest.mark.skip
    def test_init(self):
        nhead = 8
        E, R = 512, 128
        PerformerEncoderLayer(E, nhead, R)

    @pytest.mark.skip
    def test_forward(self):
        torch.random.manual_seed(42)
        nhead = 8
        N, E, R, L = 2, 512, 128, 1024
        m = PerformerEncoderLayer(E, nhead, R)
        x = torch.rand(L, N, E)
        out = m(x)
        assert isinstance(out, Tensor)

    @pytest.mark.cuda_or_skip
    def test_memory(self):
        torch.random.manual_seed(42)
        nhead = 8
        N, E, L = 2, 2048, 1024
        m = PerformerEncoderLayer(E, nhead).cuda()
        x = torch.rand(L, N, E, requires_grad=True).cuda()
        before = torch.cuda.max_memory_allocated()
        out = m(x)
        out.sum().backward()
        after = torch.cuda.max_memory_allocated()

        (after - before) / 1e6
        assert False


class TestPointPerformer:
    def test_cluster(self):
        coords = torch.rand(32, 1, 3)
        coords2 = coords[::4]
        features = torch.rand(8, 1, 1)
        PointPerformer.upsample(features, coords2, coords)
        assert False
