#!/usr/bin/env python
# -*- coding: utf-8 -*-


import timeit

import pytest
import torch

from combustion.points import projection_mask


class TestProjectionMaskFunctional:
    def test_project(self):
        torch.random.manual_seed(42)
        coords = torch.randint(0, 11, (100, 3)).float()
        mask = projection_mask(coords, 1.0)
        assert tuple(mask.shape) == (10, 10)
        assert mask.min() == -1
        assert (mask <= 100).all()

    def test_index_with_mask(self):
        torch.random.manual_seed(42)
        coords = torch.randint(0, 11, (100, 3)).float()
        mask = projection_mask(coords, 1.0)
        projected = coords[mask]
        assert tuple(projected.shape) == (10, 10, 3)

    def test_mask_trivial(self):
        coords = (
            torch.tensor([[0.0, 0.0, 0.0], [1.1, 1.1, 0.0], [1.1, 1.1, 2.0], [0.0, 10.0, 0.0], [10, 10, 0.0],])
            .float()
            .sub(5)
        )

        mask = projection_mask(coords, 1.0, (10, 10))
        assert mask[0, 0] == 0
        assert mask[9, 9] == 4
        assert mask[9, 0] == 3
        assert mask[1, 1] == 2

    @pytest.mark.parametrize("batched", [True, False])
    def test_mask_trivial_negative(self, batched):
        coords = torch.tensor(
            [[0.0, -0.5, 0.0], [-0.5, -0.5, 0.0], [0.5, 0.0, 0.0], [0.6, 0.0, 2.0], [0.75, 0.75, 2.0]]
        ).float()
        if batched:
            coords.unsqueeze_(0)

        projection_mask(coords, 1.0, (10, 10))

    @pytest.mark.parametrize("cuda", [True, False])
    def test_runtime(self, cuda):
        if cuda and not torch.cuda.is_available():
            pytest.skip(reason="CUDA not available")

        torch.random.manual_seed(42)
        coords = torch.randint(0, 1000, (1000000, 3)).float()
        coords = coords.cuda() if cuda else coords

        projection_mask(coords, 1.0, image_size=(1000, 1000))

        def func():
            projection_mask(coords, 1.0, image_size=(227, 227))

        number = 10
        t = timeit.timeit(func, number=number) / number
        assert t <= 0.1

        s = "CUDA" if cuda else "CPU"
        print(f"{s} Time: {t}")
