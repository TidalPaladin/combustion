#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
from timeit import timeit

import pytest
import torch

from combustion.nn.modules.cluster import KNNCluster, TransitionDown, TransitionUp


class TestKNNCluster:
    def test_cluster_graph(self):
        L, N, D = 5, 2, 3
        K = 3
        coords = torch.rand((L, N, 3))
        coords[:, 0, :].add_(1)
        coords[:, 1, :].sub_(1)

        features = torch.rand(L, N, D)
        features[:, 0, :].add_(1)
        features[:, 1, :].sub_(1)

        l = KNNCluster(k=K)
        indices = l(coords, coords)
        out_features = features[indices].view(K, L, N, D)
        assert tuple(out_features.shape) == (K, L, N, D)
        assert (out_features[:, :, 0, :] >= 0).all()
        assert (out_features[:, :, 1, :] <= 0).all()

        neighborhood_avg = out_features.mean(dim=0)
        assert (neighborhood_avg != features).all()

    def test_cluster(self):
        L1, N, D = 10, 2, 3
        L2 = 10
        K = 4
        coords1 = torch.arange(L1).view(L1, 1, 1).expand(-1, N, 3).contiguous()
        coords2 = coords1.neg()

        l = KNNCluster(k=K)
        indices = l(coords1, coords2)
        neighbor_coords = coords1[indices].view(K, L1, N, 3)

        expected = coords1[:K, ...].view(K, 1, N, 3).expand(-1, L2, N, 3)
        assert torch.allclose(neighbor_coords, expected)

    @pytest.mark.ci_ckip
    def test_runtime(self):
        L, N, D = 100000, 2, 32
        K = 16
        coords = torch.rand((L, N, 3))

        l = KNNCluster(k=K)
        torch.cuda.synchronize()
        t1 = time.time()
        l(coords, coords)
        torch.cuda.synchronize()
        t2 = time.time()

        t = t2 - t1
        assert t < 0.25
        assert False


class TestTransitionDown:
    def test_cluster(self):
        L, N, D = 5, 2, 3
        K = 4
        coords = torch.rand((L, N, 3))
        coords[:, 0, :].add_(1)
        coords[:, 1, :].sub_(1)

        features = torch.rand(L, N, D)
        features[:, 0, :].add_(1)
        features[:, 1, :].sub_(1)

        l = TransitionDown(D, 2 * D, K, 0.5)
        l(coords, features)
        assert False


class TestTransitionUp:
    def test_cluster(self):
        L, N, D = 5, 2, 3
        K = 4
        coords = torch.rand((L, N, 3))
        coords[:, 0, :].add_(1)
        coords[:, 1, :].sub_(1)

        features = torch.rand(L, N, D)
        features[:, 0, :].add_(1)
        features[:, 1, :].sub_(1)

        l = TransitionDown(D, 2 * D, K, 0.5)
        keep_coords, coarse, neighbor_idx, keep_idx = l(coords, features)
        l2 = TransitionUp(
            2 * D,
            D,
        )
        l2(coarse, features, neighbor_idx, keep_idx)

    @pytest.mark.ci_skip
    @pytest.mark.cuda_or_skip
    def test_runtime(self):
        L, N, D = 8192, 2, 32
        K = 16
        coords = torch.rand(L, N, 3, requires_grad=True).cuda()
        features = torch.rand(L, N, D, requires_grad=True).cuda()

        l = TransitionDown(D, 2 * D, K, 0.25).cuda()
        l2 = TransitionUp(
            2 * D,
            D,
        ).cuda()

        def func():
            keep_coords, coarse, neighbor_idx, keep_idx = l(coords, features)
            l2(coarse, features, neighbor_idx, keep_idx)

        n = 2
        t = timeit(func, number=n) / n
        assert False
