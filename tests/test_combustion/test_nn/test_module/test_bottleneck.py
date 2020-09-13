#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest

from combustion.nn import Bottleneck1d, Bottleneck2d, Bottleneck3d, BottleneckFactorized2d, BottleneckFactorized3d


@pytest.mark.parametrize(
    "conv,rank",
    [
        pytest.param(Bottleneck3d, 3, id="Bottleneck3d"),
        pytest.param(Bottleneck2d, 2, id="Bottleneck2d"),
        pytest.param(Bottleneck1d, 1, id="Bottleneck1d"),
        pytest.param(BottleneckFactorized3d, 3, id="BottleneckFactorized3d"),
        pytest.param(BottleneckFactorized2d, 2, id="BottleneckFactorized2d"),
    ],
)
def test_output_shape(torch, conv, rank):
    shape = (8,) * rank
    input = torch.ones(1, 5, *shape)
    layer = conv(5, 5, 3, bn_depth=None, bn_spatial=None, repeats=1)
    output = layer(input)
    assert output.shape == input.shape


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
@pytest.mark.parametrize(
    "conv,rank",
    [
        pytest.param(Bottleneck3d, 3, id="Bottleneck3d"),
        pytest.param(Bottleneck2d, 2, id="Bottleneck2d"),
        pytest.param(Bottleneck1d, 1, id="Bottleneck1d"),
        pytest.param(BottleneckFactorized3d, 3, id="BottleneckFactorized3d"),
        pytest.param(BottleneckFactorized2d, 2, id="BottleneckFactorized2d"),
    ],
)
def test_bottleneck(torch, conv, rank, bn_spatial, bn_depth):
    shape = (16,) * rank
    input = torch.ones(1, 16, *shape)
    layer = conv(16, 16, 3, bn_depth=bn_depth, bn_spatial=bn_spatial, repeats=1)
    output = layer(input)
    assert output.shape == input.shape


@pytest.mark.skip
@pytest.mark.parametrize("checkpoint", [True, False])
@pytest.mark.parametrize(
    "conv,rank",
    [
        pytest.param(Bottleneck3d, 3, id="Bottleneck3d"),
        pytest.param(Bottleneck2d, 2, id="Bottleneck2d"),
        pytest.param(Bottleneck1d, 1, id="Bottleneck1d"),
        pytest.param(BottleneckFactorized3d, 3, id="BottleneckFactorized3d"),
        pytest.param(BottleneckFactorized2d, 2, id="BottleneckFactorized2d"),
    ],
)
def test_checkpoint(torch, conv, rank, checkpoint, mocker):
    spy = mocker.spy(torch.utils.checkpoint, "checkpoint")
    shape = (16,) * rank
    input = torch.ones(1, 16, *shape)
    layer = conv(
        16,
        16,
        3,
        bn_depth=None,
        bn_spatial=None,
        repeats=1,
        checkpoint=checkpoint,
    )
    output = layer(input)
    if checkpoint:
        spy.assert_called()
        assert torch.allclose(output, spy.spy_return)
    else:
        spy.assert_not_called()
