#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from timeit import timeit
import torch
from combustion.nn.modules.transformer.point_transformer import PCTDown, PCTUp, ClusterModel


class TestClusterModel:

    @pytest.mark.parametrize("L", [1024, 1000])
    def test_forward(self, L):
        N, D_in = 2, 3
        C = 3
        K = 8
        D = 32
        ratio = 0.25
        features = torch.rand(L, N, D_in)
        coords = torch.rand(L, N, C)

        features[:, 0, :].add_(1)
        coords[:, 0, :].add_(1)

        features[:, 1, :].sub_(1)
        coords[:, 1, :].sub_(1)

        model = ClusterModel(D, D_in, C, blocks=[1, 1], k=K, nhead=8)

        out = model(coords, features)
        assert False


    def test_runtime_basic(self):
        L, N, D_in = 8192, 2, 3
        C = 3
        K = 16
        D = 32
        ratio = 0.25
        features = torch.rand(L, N, D_in, requires_grad=True)
        coords = torch.rand(L, N, C, requires_grad=True)

        coords = coords.cuda()
        features = features.cuda()
        model = ClusterModel(D, D_in, C, blocks=[1, 1, 1, 1], k=K, nhead=4).cuda()

        def func():
            model(coords, features)
        n = 2
        t = timeit(func, number=n) / n
        assert False









    def test_runtime(self):
        L, N, D_in = 2048, 2, 3
        C = 3
        K = 32
        D = 32
        ratio = 0.25
        features = torch.rand(L, N, D_in)
        coords = torch.rand(L, N, C)

        coords = coords
        features = features
        model = ClusterModel(D, D_in, C, blocks=[1, 1], k=K, nhead=4)
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True
        ) as p:
            model(coords, features)

        x = p.key_averages().table(sort_by="cpu_time_total", row_limit=10)
        y = p.key_averages().table(sort_by="cuda_time_total", row_limit=10)
        print(x)
        print(y)
        assert False





