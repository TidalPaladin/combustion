#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import torch

from combustion.points import center


class TestCenterFunctional:
    @pytest.mark.parametrize(
        "strategy",
        ["minmax", "mean", pytest.param("BAD", marks=pytest.mark.xfail(raises=torch.jit.Error))],  # type: ignore
    )
    def test_center_basic(self, strategy):
        torch.random.manual_seed(42)
        coords = torch.randint(5, 11, (1000, 3)).float()
        output = center(coords, strategy=strategy)
        if strategy == "minmax":
            assert torch.allclose(output.max(), torch.tensor([2.5]))
            assert torch.allclose(output.min(), torch.tensor([-2.5]))
        else:
            assert torch.allclose(output.mean(dim=0), torch.tensor([0.0, 0.0, 0.0]), atol=1e-4)

    def test_center_xy_only(self):
        torch.random.manual_seed(42)
        coords = torch.randint(5, 11, (1000, 3)).float()
        output1 = center(coords[..., :2])
        output2 = center(coords)
        assert torch.allclose(output1, output2[..., :2])

    def test_grad(self):
        torch.random.manual_seed(42)
        coords = torch.rand(1000, 3, requires_grad=True)
        output = center(coords)
        output.sum().backward()

    @pytest.mark.parametrize("dtype", [torch.float, torch.double, torch.int])
    def test_input_types(self, dtype):
        torch.random.manual_seed(42)
        coords = torch.rand(1000, 3).to(dtype)
        output = center(coords)
        assert output.is_floating_point()

        if coords.is_floating_point():
            assert coords.dtype == output.dtype

    @pytest.mark.parametrize("inplace", [True, False])
    def test_inplace(self, inplace):
        torch.random.manual_seed(42)
        coords = torch.rand(1000, 3)
        coords_copy = coords.clone()
        output = center(coords, inplace=inplace)

        if inplace:
            assert not torch.allclose(coords, coords_copy)
            assert output is coords
        else:
            assert torch.allclose(coords, coords_copy)
            assert not torch.allclose(coords, output)
