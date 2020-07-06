#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import torch
from torch.nn.functional import relu6

from combustion.nn import HardSigmoid
from combustion.testing import TorchScriptTestMixin, TorchScriptTraceTestMixin


@pytest.fixture
def input():
    torch.random.manual_seed(42)
    return torch.rand(5)


class TestHardSigmoid(TorchScriptTestMixin, TorchScriptTraceTestMixin):
    @pytest.fixture
    def model(self):
        return HardSigmoid()

    @pytest.fixture
    def data(self):
        torch.random.manual_seed(42)
        return torch.rand(5)

    @pytest.fixture
    def expected(self, input):
        return relu6(input + 3) / 6

    def test_forward(self, input, expected):
        output = HardSigmoid()(input)
        assert torch.allclose(output, expected)

    def test_backward(self, input):
        input.requires_grad = True
        layer = HardSigmoid()
        output = layer(input)
        scalar = output.sum()
        scalar.backward()

        expected_grad = torch.tensor([0.1667, 0.1667, 0.1667, 0.1667, 0.1667])
        actual_grad = input.grad

        assert torch.allclose(actual_grad, expected_grad, atol=2e-4)
