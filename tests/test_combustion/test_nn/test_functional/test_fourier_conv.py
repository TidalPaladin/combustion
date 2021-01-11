#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import torch
import torch.nn as nn
from packaging import version

from combustion.nn.functional import fourier_conv2d


has_torch18 = version.parse(torch.__version__) > version.parse("1.7.1")


@pytest.mark.skipif(not has_torch18, reason="torch>=1.8 is required")
@pytest.mark.parametrize(
    "kernel_size",
    [
        pytest.param((3, 3), id="kernel=3x3"),
        pytest.param((5, 5), id="kernel=5x5"),
        pytest.param((3, 1), id="kernel=3x1"),
    ],
)
@pytest.mark.parametrize(
    "stride",
    [
        pytest.param((1, 1), id="stride=(1, 1)"),
        pytest.param((2, 2), id="stride=(2, 2)"),
        pytest.param((2, 1), id="stride=(2, 1)"),
    ],
)
@pytest.mark.parametrize(
    "padding",
    [
        pytest.param((1, 1), id="pad=(1, 1)"),
        pytest.param((2, 2), id="pad=(2, 2)"),
        pytest.param((2, 1), id="pad=(2, 1)"),
    ],
)
@pytest.mark.parametrize(
    "bias",
    [
        pytest.param(False, id="bias=False"),
        pytest.param(True, id="bias=True"),
    ],
)
def test_conv2d(kernel_size, stride, padding, bias):
    torch.random.manual_seed(42)
    inputs = torch.rand(1, 1, 9, 9)
    kernel = torch.rand(1, 1, *kernel_size)
    if bias:
        _bias = torch.rand(1)
    else:
        _bias = None

    layer = nn.Conv2d(1, 1, kernel_size, stride, padding, bias=False)
    layer.weight = nn.Parameter(kernel)
    if bias:
        layer.bias = nn.Parameter(_bias)
    else:
        layer.register_parameter("bias", None)

    actual = fourier_conv2d(inputs, kernel, stride=stride, padding=padding, bias=_bias)
    expected = layer(inputs)
    assert actual.shape == expected.shape
    assert torch.allclose(expected, actual, atol=1e-4)
