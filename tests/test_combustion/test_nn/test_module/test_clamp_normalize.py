#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import torch

from combustion.nn import ClampAndNormalize
from combustion.testing import TorchScriptTestMixin, TorchScriptTraceTestMixin


class TestClampAndNormalize(TorchScriptTestMixin, TorchScriptTraceTestMixin):
    @pytest.fixture
    def model(self):
        return ClampAndNormalize(-10, 10)

    @pytest.fixture
    def data(self):
        return torch.tensor([10])

    @pytest.mark.parametrize(
        "inputs,min,max,norm_min,norm_max,expected",
        [
            pytest.param(3, -10, 10, 0, 1, 0.65),
            pytest.param(0, -10, 10, 0, 1, 0.5),
            pytest.param(2, 0, 10, 0, 1, 0.2),
            pytest.param(0, -10, 0, 0, 1, 1.0),
            pytest.param(0, -10, 0, -1, 1, 1.0),
            pytest.param(-10, -10, 0, -1, 1, -1.0),
            pytest.param(-10, -10, 0, -1, 0, -1.0),
            pytest.param(0, -10, 0, -1, 0, 0.0),
        ],
    )
    def test_result(self, inputs, min, max, norm_min, norm_max, expected):
        inputs = torch.tensor([inputs])
        inputs_copy = inputs.clone()
        expected = torch.tensor([expected])
        layer = ClampAndNormalize(min, max, norm_min, norm_max)
        actual = layer(inputs)
        assert torch.allclose(expected, actual)
        assert torch.allclose(inputs, inputs_copy)

    def test_validate(self):
        with pytest.raises(ValueError):
            ClampAndNormalize(10, 10)

    def test_repr(self):
        layer = ClampAndNormalize(-10, 10, -1, 2)
        print(str(layer))
