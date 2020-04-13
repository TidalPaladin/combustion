#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest

from combustion.nn import Conv1d, Conv2d, Conv3d, ConvTranspose1d, ConvTranspose2d, ConvTranspose3d


@pytest.mark.parametrize(
    "conv,rank",
    [pytest.param(Conv3d, 3, id="Conv3d"), pytest.param(Conv2d, 2, id="Conv2d"), pytest.param(Conv1d, 1, id="Conv1d"),],
)
def test_conv(torch, conv, rank):
    shape = (8,) * rank
    input = torch.ones(1, 1, *shape)
    layer = conv(1, 5, 3, padding=1)
    output = layer(input)
    assert output.shape == (1, 5, *shape)


@pytest.mark.parametrize(
    "conv,rank",
    [
        pytest.param(ConvTranspose3d, 3, id="ConvTranspose3d"),
        pytest.param(ConvTranspose2d, 2, id="ConvTranspose2d"),
        pytest.param(ConvTranspose1d, 1, id="ConvTranspose1d"),
    ],
)
def test_transpose_conv(torch, conv, rank):
    shape = (8,) * rank
    input = torch.ones(1, 1, *shape)
    layer = conv(1, 5, 3, padding=1)
    output = layer(input)
    assert output.shape == (1, 5, *shape)


@pytest.mark.parametrize(
    "conv,rank",
    [pytest.param(Conv3d, 3, id="Conv3d"), pytest.param(Conv2d, 2, id="Conv2d"), pytest.param(Conv1d, 1, id="Conv1d"),],
)
def test_conv_pad(torch, conv, rank):
    shape = (8,) * rank
    out_shape = (6,) * rank
    input = torch.ones(1, 1, *shape)
    layer = conv(1, 5, 3, padding=0)
    output = layer(input)
    assert output.shape == (1, 5, *out_shape)


@pytest.mark.parametrize(
    "conv,rank",
    [
        pytest.param(ConvTranspose3d, 3, id="ConvTranspose3d"),
        pytest.param(ConvTranspose2d, 2, id="ConvTranspose2d"),
        pytest.param(ConvTranspose1d, 1, id="ConvTranspose1d"),
    ],
)
def test_transpose_conv_pad(torch, conv, rank):
    shape = (8,) * rank
    out_shape = (10,) * rank
    input = torch.ones(1, 1, *shape)
    layer = conv(1, 5, 3, padding=0)
    output = layer(input)
    assert output.shape == (1, 5, *out_shape)


@pytest.mark.parametrize(
    "conv,rank",
    [pytest.param(Conv3d, 3, id="Conv3d"), pytest.param(Conv2d, 2, id="Conv2d"), pytest.param(Conv1d, 1, id="Conv1d"),],
)
def test_conv_stride(torch, conv, rank):
    shape = (8,) * rank
    out_shape = (4,) * rank
    input = torch.ones(1, 1, *shape)
    layer = conv(1, 5, 3, padding=1, stride=2)
    output = layer(input)
    assert output.shape == (1, 5, *out_shape)


@pytest.mark.parametrize(
    "conv,rank",
    [
        pytest.param(ConvTranspose3d, 3, id="ConvTranspose3d"),
        pytest.param(ConvTranspose2d, 2, id="ConvTranspose2d"),
        pytest.param(ConvTranspose1d, 1, id="ConvTranspose1d"),
    ],
)
def test_transpose_conv_stride(torch, conv, rank):
    shape = (8,) * rank
    out_shape = (16,) * rank
    input = torch.ones(1, 1, *shape)
    layer = conv(1, 5, 2, padding=0, stride=2)
    output = layer(input)
    assert output.shape == (1, 5, *out_shape)
