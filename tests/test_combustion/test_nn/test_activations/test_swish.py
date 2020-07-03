#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import torch
from torch.nn.functional import relu6

from combustion.nn import HardSwish, Swish
from combustion.testing import TorchScriptTestMixin, TorchScriptTraceTestMixin


@pytest.fixture
def input():
    torch.random.manual_seed(42)
    return torch.rand(5)


class TestSwish(TorchScriptTraceTestMixin):
    @pytest.fixture
    def model(self):
        return Swish()

    @pytest.fixture
    def data(self):
        torch.random.manual_seed(42)
        return torch.rand(5)

    @pytest.fixture
    def expected(self, input):
        return input * torch.sigmoid(input)

    def test_forward(self, input, expected):
        output = Swish()(input)
        assert torch.allclose(output, expected)

    def test_backward(self, input):
        input.requires_grad = True
        layer = Swish()
        output = layer(input)
        scalar = output.sum()
        scalar.backward()

        expected_grad = torch.tensor([0.8899, 0.9009, 0.6869, 0.9151, 0.6904])
        actual_grad = input.grad

        assert torch.allclose(actual_grad, expected_grad, atol=2e-4)


class TestHardSwish(TorchScriptTestMixin, TorchScriptTraceTestMixin):
    @pytest.fixture
    def model(self):
        return HardSwish()

    @pytest.fixture
    def data(self):
        torch.random.manual_seed(42)
        return torch.rand(5)

    @pytest.fixture
    def expected(self, input):
        return input * relu6(input + 3) / 6

    @pytest.mark.parametrize("inplace", [True, False])
    def test_forward(self, input, expected, inplace):
        original_input = input.clone()
        output = HardSwish(inplace=inplace)(input)
        assert torch.allclose(output, expected)

        if inplace:
            assert output is input
        else:
            assert torch.allclose(input, original_input)

    def test_backward(self, input):
        input.requires_grad = True
        layer = HardSwish()
        output = layer(input)
        scalar = output.sum()
        scalar.backward()

        expected_grad = torch.tensor([0.7941, 0.8050, 0.6276, 0.8198, 0.6301])
        actual_grad = input.grad

        assert torch.allclose(actual_grad, expected_grad, atol=2e-4)
