#!/usr/bin/env python
# -*- coding: utf-8 -*-

from math import pi

import pytest
import torch

from combustion.nn.functional import cartesian_to_polar, polar_to_cartesian


def get_params():
    return [
        pytest.param(0.5, 0.5, 0.7071, pi / 4),
        pytest.param(-0.5, 0.5, 0.7071, 3 * pi / 4),
        pytest.param(0.5, -0.5, 0.7071, -pi / 4),
        pytest.param(1, 1, 1.4142, pi / 4),
        pytest.param(0, 0, 0, pi),
    ]


class TestCartesianToPolar:
    @pytest.mark.parametrize("x,y,r,t", get_params())
    def test_compute(self, x, y, r, t):
        x = torch.tensor(x)
        y = torch.tensor(y)
        r = torch.tensor(r)
        t = torch.tensor(t)

        r_computed, t_computed = cartesian_to_polar(x, y)
        assert torch.allclose(r_computed, r.float(), atol=1e-4)
        assert torch.allclose(t_computed, t.float(), atol=1e-4)


class TestPolarToCartesian:
    @pytest.mark.parametrize("x,y,r,t", get_params())
    def test_compute(self, x, y, r, t):
        x = torch.tensor(x)
        y = torch.tensor(y)
        r = torch.tensor(r)
        t = torch.tensor(t)

        x_computed, y_computed = polar_to_cartesian(r, t)
        assert torch.allclose(x_computed, x.float(), atol=1e-4)
        assert torch.allclose(y_computed, y.float(), atol=1e-4)
