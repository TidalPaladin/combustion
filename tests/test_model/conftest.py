#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gc

import pytest


@pytest.fixture(
    params=[pytest.param(3, id="dim=3"), pytest.param(9, id="dim=9"),]
)
def frames(request):
    return request.param


@pytest.fixture(
    params=[pytest.param((64, 64), id="image=64x64"), pytest.param((32, 32), id="image=32x32"),]
)
def image_size(request):
    return request.param


@pytest.fixture(
    params=[pytest.param(2, id="batch=2"),]
)
def batch_size(request):
    return request.param


@pytest.fixture(
    params=[pytest.param(8, id="width=8"), pytest.param(16, id="width=16"),]
)
def width(request):
    return request.param


@pytest.fixture
def input(torch, frames, image_size, batch_size, width):
    result = torch.rand(batch_size, 1, frames, *image_size)
    yield result
    del result
    gc.collect()


@pytest.fixture(
    params=[pytest.param(1, id="upsample=1x"), pytest.param(2, id="upsample=2x"), pytest.param(4, id="upsample=4x"),]
)
def upsample(request):
    return request.param


@pytest.fixture
def output_shape(batch_size, image_size, upsample):
    out_size = tuple([x * upsample for x in image_size])
    return tuple(batch_size, 1, *out_size)


@pytest.fixture
def window(mock_args, frames):
    if frames == 9:
        mock_args.dense_window = 4
        mock_args.sparse_window = 0
    else:
        mock_args.dense_window = 0
        mock_args.sparse_window = 4
    return frames


@pytest.fixture(params=["mse", "bce", "wmse", "wbce", "focal"])
def criterion(request, mock_args):
    mock_args.criterion = request.param
    return request.param


@pytest.fixture
def head(mocker, torch):
    m = mocker.MagicMock(spec_set=torch.nn.Module, name="head")
    mocker.patch("combustion.model.baseline.RegressionHead3D", m)
    mocker.patch("combustion.model.unet.RegressionHead3D", m)
    mocker.patch("combustion.model.baseline.ClassificationHead3D", m)
    mocker.patch("combustion.model.unet.ClassificationHead3D", m)
    mocker.patch("combustion.model.efficient.RegressionHead2D", m)
    mocker.patch("combustion.model.efficient.ClassificationHead2D", m)
    return m
