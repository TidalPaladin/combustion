#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pytest
from timeit import timeit
import torch
from pathlib import Path
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

    @pytest.mark.ci_skip
    def test_plot(self):
        path = Path("/home/tidal/test_imgs/TestClusterModel")
        if not path.is_dir():
            return

        L, N, D_in = 1024, 2, 3
        C = 2
        K = 16
        D = 32
        ratio = 0.25
        features = torch.randn(L, N, D_in, requires_grad=True)
        x = torch.rand(L, N, 1, requires_grad=False)
        y = torch.randn(L, N, 1, requires_grad=False)
        coords = torch.cat((x, y), dim=-1)

        coords = coords
        features = features
        model = ClusterModel(D, D_in, C, blocks=[1, 1, 1], k=K, nhead=4)
        model(coords, features)

        enc_coords = [l.coords for l in model.encoder]
        dec_coords = [l.coords for l in model.decoder]

        for i, c in enumerate(enc_coords):
            name = Path(path, f"enc_{i}.png")
            plt.scatter(c[..., 0], c[..., 1])
            plt.savefig(str(name))
            plt.close()

        for i, c in enumerate(dec_coords):
            name = Path(path, f"dec_{i}.png")
            plt.scatter(c[..., 0], c[..., 1])
            plt.savefig(str(name))
            plt.close()

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





