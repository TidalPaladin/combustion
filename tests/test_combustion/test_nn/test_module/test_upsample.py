#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest

from combustion.nn import UpSample3d


pytest.skip(allow_module_level=True)


@pytest.fixture
def input(torch):
    return torch.ones(1, 6, 8, 8, 8)


@pytest.fixture(
    params=[
        pytest.param(True, id="factorized"),
        pytest.param(False, id="nonfactorized"),
    ]
)
def factorized(request):
    return request.param


def test_output_shape(torch, input, factorized):
    expected_shape = (1, 3, 16, 16, 16)
    layer = UpSample3d(6, 3, 3, 2, factorized=factorized, stride=2, padding=0)
    output, _ = layer(input)
    assert output.shape == expected_shape


def test_returns_pre_final_output(torch, input, factorized):
    layer = UpSample3d(6, 3, 3, 2, factorized=factorized, stride=2, padding=0)
    _, main = layer(input)
    assert main.shape == input.shape
    assert not (main is input)


def test_stride(torch, input, factorized):
    expected_shape = (1, 3, 9, 9, 9)
    layer = UpSample3d(6, 3, 3, 2, factorized=factorized, stride=1, padding=0)
    output, _ = layer(input)
    assert output.shape == expected_shape


def test_pad(torch, input, factorized):
    expected_shape = (1, 3, 7, 7, 7)
    layer = UpSample3d(6, 3, 3, 2, factorized=factorized, stride=1, padding=1)
    output, _ = layer(input)
    assert output.shape == expected_shape


def test_repeats(torch, input, factorized):
    expected_shape = (1, 3, 16, 16, 16)
    layer1 = UpSample3d(6, 3, 3, 2, factorized=factorized, stride=2, padding=0, repeats=1)
    layer2 = UpSample3d(6, 3, 3, 2, factorized=factorized, stride=2, padding=0, repeats=2)
    output, _ = layer2(input)
    assert len(list(layer2.parameters())) > len(list(layer1.parameters()))
    assert output.shape == expected_shape


@pytest.mark.parametrize(
    "bn_spatial",
    [
        pytest.param(None, id="bn_spatial=None"),
        pytest.param(2, id="bn_spatial=2"),
        pytest.param(4, id="bn_spatial=4"),
    ],
)
@pytest.mark.parametrize(
    "bn_depth",
    [
        pytest.param(None, id="bn_depth=None"),
        pytest.param(2, id="bn_depth=2"),
        pytest.param(4, id="bn_depth=4"),
    ],
)
def test_bottleneck(torch, input, factorized, bn_depth, bn_spatial):
    expected_shape = (1, 3, 16, 16, 16)
    layer = UpSample3d(
        6,
        3,
        3,
        2,
        factorized=factorized,
        stride=2,
        padding=0,
        repeats=1,
        bn_spatial=bn_spatial,
        bn_depth=bn_depth,
    )
    output, _ = layer(input)
    assert output.shape == expected_shape
