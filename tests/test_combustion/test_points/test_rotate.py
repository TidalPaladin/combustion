#!/usr/bin/env python
# -*- coding: utf-8 -*-

import timeit
from math import radians

import pytest
import torch

from combustion.points import RandomRotate, Rotate, random_rotate, rotate


class TestRotateFunctional:
    @pytest.mark.parametrize(
        "x,y,z",
        [
            pytest.param(0.0, 0.0, 0.0, id="x=y=z=0"),
            pytest.param(radians(360), radians(360), radians(360), id="x=y=z=360"),
        ],
    )
    @pytest.mark.parametrize("batched", [pytest.param(True, id="batched"), pytest.param(False, id="unbatched")])
    def test_rotate_complete_rotations(self, x, y, z, batched):
        if batched:
            coords = torch.rand(1, 10, 3)
        else:
            coords = torch.rand(10, 3)
        output = rotate(coords, x, y, z)
        assert output.shape == coords.shape
        assert torch.allclose(coords, output)

    def test_input_unchanged(self):
        x, y, z = (0.5,) * 3
        coords = torch.rand(1, 10, 3)
        original_coords = coords.clone()
        rotate(coords, x, y, z)
        assert coords.shape == original_coords.shape
        assert torch.allclose(coords, original_coords)

    @pytest.mark.parametrize("batched", [pytest.param(True, id="batched"), pytest.param(False, id="unbatched")])
    def test_rotate_point_at_origin(self, batched):
        x, y, z, = 0.5, 0.23, 0.1
        if batched:
            coords = torch.zeros(1, 10, 3).float()
        else:
            coords = torch.zeros(10, 3).float()
        output = rotate(coords, x, y, z)
        assert output.shape == coords.shape
        assert torch.allclose(coords, output)

    @pytest.mark.skip
    @pytest.mark.parametrize("batched", [pytest.param(True, id="batched"), pytest.param(False, id="unbatched")])
    @pytest.mark.parametrize("degrees", [pytest.param(True, id="degrees"), pytest.param(False, id="radians")])
    @pytest.mark.parametrize(
        "x,y,z",
        [
            pytest.param(180, 0, 0, id="x=180"),
            pytest.param(0, 180, 0, id="y=180"),
            pytest.param(0, 0, 180, id="z=180"),
            pytest.param(180, 180, 180, id="x=y=z=180"),
        ],
    )
    def test_rotated_points(self, x, y, z, batched, degrees):
        coords = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1],]).float()

        if not degrees:
            x, y, z = radians(x), radians(y), radians(z)

        output = rotate(coords, x, y, z)
        assert torch.allclose(coords, output)

    @pytest.mark.parametrize("cuda", [True, False])
    def test_runtime(self, cuda):
        if cuda and not torch.cuda.is_available():
            pytest.skip(reason="CUDA not available")

        torch.random.manual_seed(42)
        coords = torch.randint(0, 1000, (10000000, 3)).float()
        coords = coords.cuda() if cuda else coords

        rotate(coords, 1.0, 1.0, 1.0)

        def func():
            rotate(coords, 1.0, 1.0, 1.0)

        number = 50
        t = timeit.timeit(func, number=number) / number
        assert t <= 0.1

        s = "CUDA" if cuda else "CPU"
        print(f"{s} Time: {t}")

    def test_dtype(self):
        torch.random.manual_seed(42)
        coords = torch.randint(0, 1000, (10000000, 3)).double()
        output = rotate(coords, 1.0, 1.0, 1.0)
        assert output.dtype == coords.dtype


class TestRotateModule:
    def test_rotate_module(self):
        x, y, z = (0.4, 0.5, 0.6)
        xform = Rotate(x, y, z)
        coords = torch.rand(1, 10, 3)
        expected = rotate(coords, x, y, z)
        actual = xform(coords)
        assert torch.allclose(expected, actual)

    def test_module_repr(self):
        x, y, z = (0.4, 0.5, 0.6)
        xform = Rotate(x, y, z)
        print(xform)


class TestRandomRotateFunctional:
    @pytest.mark.parametrize("batched", [pytest.param(True, id="batched"), pytest.param(False, id="unbatched")])
    def test_random_rotate_zero_range(self, batched):
        if batched:
            coords = torch.rand(1, 10, 3)
        else:
            coords = torch.rand(10, 3)

        x, y, z = ((0.0, 0.0),) * 3
        output = random_rotate(coords, x, y, z)
        assert output.shape == coords.shape
        assert torch.allclose(coords, output)

    def test_random_rotation(self):
        x, y, z = ((-0.5, 0.5),) * 3
        coords = torch.rand(1, 10, 3)
        out1 = random_rotate(coords, x, y, z)
        out2 = random_rotate(coords, x, y, z)
        assert not torch.allclose(out1, out2)

    @pytest.mark.parametrize(
        "x,y,z",
        [
            pytest.param((1, -1), (0.0, 0.0), (0.0, 0.0), id="x"),
            pytest.param((0.0, 0.0), (1, -1), (0.0, 0.0), id="y"),
            pytest.param((0.0, 0.0), (0.0, 0.0), (1, -1), id="z"),
            pytest.param((0.0, 0.0, 0.0), (0.0, 0.0), (0.0, 0.0), id="length"),
            pytest.param("foo", "bar", "baz", id="type"),
        ],
    )
    def test_validate_input(self, x, y, z):
        coords = torch.rand(1, 10, 3)
        with pytest.raises(ValueError):
            random_rotate(coords, x, y, z)

    def test_input_unchanged(self):
        x, y, z = ((-0.5, 0.5),) * 3
        coords = torch.rand(1, 10, 3)
        original_coords = coords.clone()
        random_rotate(coords, x, y, z)
        assert coords.shape == original_coords.shape
        assert torch.allclose(coords, original_coords)

    @pytest.mark.parametrize("batched", [pytest.param(True, id="batched"), pytest.param(False, id="unbatched")])
    def test_rotate_point_at_origin(self, batched):
        x, y, z, = ((0.0, 1.0),) * 3
        if batched:
            coords = torch.zeros(1, 10, 3).float()
        else:
            coords = torch.zeros(10, 3).float()
        output = random_rotate(coords, x, y, z)
        assert output.shape == coords.shape
        assert torch.allclose(coords, output)

    @pytest.mark.parametrize("batched", [pytest.param(True, id="batched"), pytest.param(False, id="unbatched")])
    @pytest.mark.parametrize("degrees", [pytest.param(True, id="degrees"), pytest.param(False, id="radians")])
    @pytest.mark.parametrize(
        "x,y,z",
        [
            pytest.param(80, 0, 0, id="x=80"),
            pytest.param(0, 80, 0, id="y=80"),
            pytest.param(0, 0, 80, id="z=80"),
            pytest.param(80, 80, 80, id="x=y=z=80"),
        ],
    )
    def test_matches_rotate_fixed_range(self, x, y, z, batched, degrees):
        coords = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1],]).float()
        if not degrees:
            x, y, z = radians(x), radians(y), radians(z)
        expected = rotate(coords, x, y, z, degrees)

        x, y, z = [(val, val) for val in (x, y, z)]
        actual = random_rotate(coords, x, y, z, degrees)
        assert torch.allclose(expected, actual)


class TestRandomRotateModule:
    def test_rotate_module(self):
        x, y, z = ((-0.5, 0.5),) * 3
        xform = RandomRotate(x, y, z)
        coords = torch.rand(1, 10, 3)
        xform(coords)

    def test_module_repr(self):
        x, y, z = ((-0.1, 0.1),) * 3
        xform = RandomRotate(x, y, z)
        print(xform)

    @pytest.mark.parametrize(
        "x,y,z",
        [
            pytest.param((1, -1), (0.0, 0.0), (0.0, 0.0), id="x"),
            pytest.param((0.0, 0.0), (1, -1), (0.0, 0.0), id="y"),
            pytest.param((0.0, 0.0), (0.0, 0.0), (1, -1), id="z"),
            pytest.param((0.0, 0.0, 0.0), (0.0, 0.0), (0.0, 0.0), id="length"),
            pytest.param("foo", "bar", "baz", id="type"),
        ],
    )
    def test_validate_input(self, x, y, z):
        with pytest.raises(ValueError):
            RandomRotate(x, y, z)
