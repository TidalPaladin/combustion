#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import pytest
from timeit import timeit

from combustion.nn.modules.cluster import KNNCluster, InverseCluster, PointReduction, PointPooling, TransitionDown, TransitionUp

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
        features = torch.rand(L, N, D)

        l = KNNCluster(k=K)
        def func():
            l(coords, features)

        ncalls = 3
        t = timeit(func, number=ncalls) / ncalls
        assert t < 0.25

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

        l = TransitionDown(D, 2*D, K, 0.5)
        out = l(coords, features)
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

        l = TransitionDown(D, 2*D, K, 0.5)
        keep_coords, coarse, neighbor_idx, keep_idx = l(coords, features)
        l2 = TransitionUp(2*D, D, )
        final_features = l2(coarse, features, neighbor_idx, keep_idx)

    @pytest.mark.ci_skip
    def test_runtime(self):
        L, N, D = 8192, 2, 32
        K = 16
        coords = torch.rand(L, N, 3, requires_grad=True).cuda()
        features = torch.rand(L, N, D, requires_grad=True).cuda()

        l = TransitionDown(D, 2*D, K, 0.25).cuda()
        l2 = TransitionUp(2*D, D, ).cuda()

        def func():
            keep_coords, coarse, neighbor_idx, keep_idx = l(coords, features)
            final_features = l2(coarse, features, neighbor_idx, keep_idx)
        n = 2
        t = timeit(func, number=n) / n
        assert False


class TestFarthestPointsSampling:

    @pytest.mark.skip
    def test_cluster(self):
        L, N, D = 5, 2, 3
        K = 4
        coords = torch.rand((L, N, 3))
        coords[:, 0, :].add_(1)
        coords[:, 1, :].sub_(1)

        features = torch.rand(L, N, D)
        features[:, 0, :].add_(1)
        features[:, 1, :].sub_(1)
        features.requires_grad = True

        l = FarthestPointCluster(ratio=0.5)
        l = FastFarthestPointSampling(ratio=0.5)
        out_features, keep, reverse_idx = l(coords, features)
        rev = InverseCluster()
        upsampled = rev(out_features, features, reverse_idx)


    @pytest.mark.ci_ckip
    def test_runtime(self):
        L, N, D = 10000, 2, 32
        coords = torch.rand((L, N, 3))
        features = torch.rand(L, N, D)

        l = FarthestPointSampling(ratio=0.25)
        def func():
            l(coords, features)

        ncalls = 3
        t = timeit(func, number=ncalls) / ncalls
        assert t < 0.25

class TestVoxelDownsample:

    def test_cluster(self):
        L, N, D = 5, 2, 3
        K = 4
        coords = torch.rand((L, N, 3))
        coords[:, 0, :].add_(1)
        coords[:, 1, :].sub_(1)

        features = torch.rand(L, N, D)
        features[:, 0, :].add_(1)
        features[:, 1, :].sub_(1)
        features.requires_grad = True

        c = KNNCluster(k=K)
        out_features, indices = c(coords, features)

        l = PointPooling(ratio=0.5)
        out_features, keep, nearest, new_coords = l(coords, features, indices)
        final_features = out_features[nearest].view_as(features) + features
        assert False
