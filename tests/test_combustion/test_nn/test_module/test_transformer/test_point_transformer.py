#!/usr/bin/env python
# -*- coding: utf-8 -*-


import torch

from combustion.nn.modules.transformer.point_transformer import PointTransformer


class TestPointTransformer:
    def test_forward(self):
        torch.random.manual_seed(42)
        L, N, C = 1024, 2, 3
        D = 32
        coords = torch.rand(L, N, C)
        features = torch.rand(L, N, D)

        model = PointTransformer(D, repeats=[1, 1, 2])
        indices = model.get_indices(coords, k=8)
        out = model(features, indices)
        assert out.shape == features.shape
