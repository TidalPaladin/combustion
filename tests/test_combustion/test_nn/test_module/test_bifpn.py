#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from combustion.nn import BiFPN
from combustion.testing import TorchScriptTestMixin, cuda_or_skip


def custom_conv(num_channels) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(num_channels, num_channels, 3, padding=1), nn.ReLU())


@pytest.fixture(params=[5, 3, 2])
def levels(request):
    return request.param


@pytest.fixture(params=[32, 64, 16])
def num_channels(request):
    return request.param


@pytest.fixture(params=[None, custom_conv])
def conv(request):
    return request.param


@pytest.fixture(params=[1, 4])
def inputs(request, levels, num_channels):
    batch_size = request.param
    base_size = 8
    result = []
    for i in range(levels):
        height, width = (base_size * 2 ** i,) * 2
        t = torch.rand(batch_size, num_channels, height, width)
        result.append(t)
    return list(reversed(result))


def test_init(levels, num_channels, conv):
    BiFPN(num_channels, levels, conv)


@pytest.mark.parametrize(
    "levels,num_channels,conv,epsilon",
    [
        pytest.param(0, 32, None, 1e-4, id="levels=0"),
        pytest.param(2, 0, None, 1e-4, id="num_channels=0"),
        pytest.param(2, 32, 20, 1e-4, id="conv_uncallable"),
        pytest.param(2, 32, None, 0.0, id="epsilon=0"),
    ],
)
def test_validation(levels, num_channels, conv, epsilon):
    with pytest.raises(ValueError):
        BiFPN(num_channels, levels, conv, epsilon)


@cuda_or_skip
def test_forward(levels, num_channels, conv, inputs):
    layer = BiFPN(num_channels, levels, conv).cuda()
    output = layer([x.cuda() for x in inputs])
    assert isinstance(output, list)
    for out_item, in_item in zip(output, inputs):
        assert isinstance(out_item, Tensor)
        assert out_item.shape == in_item.shape
        assert out_item.requires_grad


class TestScriptable(TorchScriptTestMixin):
    @pytest.fixture
    def model(self):
        return BiFPN(32, 5)


@cuda_or_skip
def test_known_input():
    torch.random.manual_seed(42)
    layer = BiFPN(2, 3)
    input = [
        torch.rand(1, 2, 16, 16),
        torch.rand(1, 2, 8, 8),
        torch.rand(1, 2, 4, 4),
    ]
    output = layer(input)

    # outout maps are large, so just compare the sums of elements in each level
    output_sum = [x.sum() for x in output]
    expected_output_sum = [torch.tensor(205.5710469), torch.tensor(51.3697109), torch.tensor(12.5726123)]

    for level, (expected, actual) in enumerate(zip(expected_output_sum, output_sum)):
        assert torch.allclose(expected, actual, rtol=0.1)
