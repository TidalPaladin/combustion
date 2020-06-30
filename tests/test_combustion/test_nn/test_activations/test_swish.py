#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import torch

from combustion.nn import Swish
from combustion.testing import TorchScriptTraceTestMixin


@pytest.fixture
def input():
    torch.random.manual_seed(42)
    return torch.rand(5)


@pytest.fixture
def expected(input):
    return input * torch.sigmoid(input)


def test_forward(input, expected):
    output = Swish()(input)
    assert torch.allclose(output, expected)


def test_backward(input):
    input.requires_grad = True
    layer = Swish()
    output = layer(input)
    scalar = output.sum()
    scalar.backward()

    expected_grad = torch.tensor([0.8899, 0.9009, 0.6869, 0.9151, 0.6904])
    actual_grad = input.grad

    assert torch.allclose(actual_grad, expected_grad, atol=2e-4)


class TestTrace(TorchScriptTraceTestMixin):
    @pytest.fixture
    def model(self):
        return Swish()

    @pytest.fixture
    def data(self):
        torch.random.manual_seed(42)
        return torch.rand(5)
