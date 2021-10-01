#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import torch

from combustion.nn.modules.cluster import Indices, KNNCluster, NearestCluster, RandomDecimate
from combustion.util import MISSING


class TestKNNCluster:
    @pytest.mark.parametrize("k", [1, 3, 5])
    def test_cluster(self, k):
        coords = (
            torch.tensor(
                [
                    (0, 0, 0),
                    (1, 0, 0),
                    (0, 1, 0),
                    (0, 0, 1),
                    (1, 1, 0),
                    (0, 1, 1),
                    (1, 0, 1),
                    (1, 1, 1),
                ]
            )
            .float()
            .unsqueeze(1)
        )
        L, N, C = coords.shape

        cluster = KNNCluster(k=k)
        indices = cluster(coords, coords)
        indexed = coords[indices].view(k, L, N, C)
        assert torch.allclose(indexed[0], coords)

    def test_preserves_batch(self):
        L, N, C = 1024, 2, 3
        K = 8
        coords = torch.rand(L, N, C)
        coords[:, 0, :].add_(1)
        cluster = KNNCluster(k=K)
        indices = cluster(coords, coords)
        indexed = coords[indices].view(K, L, N, C)
        assert (indexed[:, :, 0, :] >= 1).all()
        assert (indexed[:, :, 1, :] < 1).all()


def test_nearest_cluster_same():
    coords = (
        torch.tensor(
            [
                (0, 0, 0),
                (1, 0, 0),
                (0, 1, 0),
                (0, 0, 1),
                (1, 1, 0),
                (0, 1, 1),
                (1, 0, 1),
                (1, 1, 1),
            ]
        )
        .float()
        .unsqueeze(1)
    )
    L, N, C = coords.shape

    cluster = NearestCluster()
    indices = cluster(coords, coords)
    indexed = coords[indices].view(L, N, C)
    assert torch.allclose(indexed, coords)


def test_nearest_cluster_different():
    coords = (
        torch.tensor(
            [
                (0, 0, 0),
                (1, 0, 0),
                (0, 1, 0),
                (0, 0, 1),
                (1, 1, 0),
                (0, 1, 1),
                (1, 0, 1),
                (1, 1, 1),
            ]
        )
        .float()
        .unsqueeze(1)
    )
    coords2 = (
        torch.tensor(
            [
                (2, 2, 2),
                (-1, -1, -1),
            ]
        )
        .float()
        .unsqueeze(1)
    )
    L, N, C = coords.shape

    cluster = NearestCluster()
    indices = cluster(coords2, coords)
    indexed = coords2[indices].view(L, N, C)
    expected = (
        torch.tensor(
            [
                (-1, -1, -1),
                (-1, -1, -1),
                (-1, -1, -1),
                (-1, -1, -1),
                (2, 2, 2),
                (2, 2, 2),
                (2, 2, 2),
                (2, 2, 2),
            ]
        )
        .type_as(indexed)
        .unsqueeze(1)
    )
    assert torch.allclose(indexed, expected)


class TestIndices:
    def test_create_indices(self):
        coords = (
            torch.tensor(
                [
                    (0, 0, 0),
                    (1, 0, 0),
                    (0, 1, 0),
                    (0, 0, 1),
                    (1, 1, 0),
                    (0, 1, 1),
                    (1, 0, 1),
                    (1, 1, 1),
                ]
            )
            .float()
            .unsqueeze(1)
        )
        L, N, C = coords.shape

        knn = KNNCluster(k=3)
        down = RandomDecimate(0.25)
        up = NearestCluster()

        idx = Indices.create(coords, knn, down, up)
        assert idx.knn is not MISSING
        assert idx.downsample is not MISSING
        assert idx.upsample is not MISSING

    def test_apply_knn(self):
        coords = (
            torch.tensor(
                [
                    (0, 0, 0),
                    (1, 0, 0),
                    (0, 1, 0),
                    (0, 0, 1),
                    (1, 1, 0),
                    (0, 1, 1),
                    (1, 0, 1),
                    (1, 1, 1),
                ]
            )
            .float()
            .unsqueeze(1)
        )
        L, N, C = coords.shape

        knn = KNNCluster(k=3)

        idx = Indices.create(coords, knn)
        out = idx.apply_knn(coords)
        assert torch.allclose(out[0], coords)

    def test_apply_knn_preserves_batch(self):
        L, N, C = 1024, 2, 3
        coords = torch.rand(L, N, C)
        coords[:, 0, :].add_(1)

        knn = KNNCluster(k=3)
        idx = Indices.create(coords, knn)
        out = idx.apply_knn(coords)
        assert (out[..., 0, :] >= 1).all()
        assert (out[..., 1, :] < 1).all()

    def test_unbatch(self):
        L, N, D = 1024, 2, 6
        torch.rand(L, N, D)
        idx = torch.randint(0, 1024, (L, N))
        out = Indices.unbatch_indices(idx, dim=1)

    def test_unbatch_knn(self):
        L, N, D = 1024, 2, 6
        K = 3
        coords = torch.rand(L, N, D)
        knn = KNNCluster(k=K)
        out = Indices.create(coords, knn)
        expected = out.apply_knn(coords)

        _knn = out.knn[0].view(-1, 2)
        result = Indices.unbatch_indices(_knn, dim=1)
        actual = coords[result].view(K, L, N, -1)

        assert torch.allclose(actual, expected)
