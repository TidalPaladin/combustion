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

    @pytest.mark.skip
    @pytest.mark.parametrize(
        "padding_mode", ["reflect", pytest.param("replicate", marks=pytest.mark.xfail(raises=NotImplementedError))]
    )
    def test_padding_value(self, padding_mode):
        if padding_mode == "replicate":
            pytest.skip("not implemented")
        torch.random.manual_seed(42)
        coords = torch.randint(-1, 2, (1000, 3)).float()
        new = torch.tensor([[2, 1, 0], [1, 2, 0],]).type_as(coords)
        coords = torch.cat([coords, new], dim=0)

        mask = projection_mask(coords, 1.0, (5, 5), padding_mode=padding_mode)
        assert (mask != -1).all()

    @pytest.mark.parametrize("cuda", [True, False])
    def test_runtime(self, cuda):
        if cuda and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

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

    @pytest.mark.parametrize("size", [4, 5])
    def test_circular_projection(self, size):
        torch.random.manual_seed(42)
        coords = torch.randint(-2 * size, 2 * size + 1, (10000, 3)).float()
        x, y, z = coords[..., 0], coords[..., 1], coords[..., 2]
        outside_circle = x ** 2 + y ** 2 > size ** 2

        grid_x, grid_y = torch.meshgrid(torch.arange(-size, size + 1), torch.arange(-size, size + 1))
        expected_within_circle = grid_x ** 2 + grid_y ** 2 <= size ** 2

        coords = coords[~outside_circle]
        mask = projection_mask(coords, 1.0, (2 * size + 1, 2 * size + 1))
        assert (expected_within_circle == (mask != -1)).all()

    @pytest.mark.parametrize("resolution", [0.5, 1.0, 2.0])
    def test_cropping(self, resolution):
        torch.random.manual_seed(42)
        size = 10
        coords = torch.randint(-2 * size, 2 * size + 1, (10000, 3)).float()
        x, y, z = coords[..., 0], coords[..., 1], coords[..., 2]

        coords2 = coords.clone()
        smaller = (x.abs() <= size / 2) & (y.abs() <= size / 2)
        coords2 = coords2[smaller]

        mask1 = projection_mask(coords, resolution, (size, size))
        mask2 = projection_mask(coords2, resolution, (size, size))

        out1 = coords[mask1]
        out1[mask1 == -1] = 0
        out2 = coords2[mask2]
        out2[mask2 == -1] = 0

        assert torch.allclose(out1, out2)
