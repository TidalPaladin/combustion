#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import torch
from torch.distributions import Binomial

from combustion.testing import cuda_or_skip
from combustion.vision import PointsToAnchors


@pytest.fixture(params=[None, 1, 2])
def batch_size(request):
    return request.param


@pytest.fixture(params=[10, 20])
def max_rois(request):
    return request.param


@pytest.fixture(params=[(32, 32), (128, 64)])
def input_size(request):
    return request.param


@pytest.fixture(params=[2, 4])
def upsample(request):
    return request.param


@pytest.fixture(params=[2, 4])
def num_classes(request):
    return request.param


@pytest.fixture(params=[0.0, 0.5])
def threshold(request):
    return request.param


@pytest.fixture
def one_hot_indices(batch_size, num_classes, input_size, max_rois):
    torch.random.manual_seed(42)
    height, width = input_size

    if batch_size is not None:
        out_shape = (batch_size, num_classes, height, width)
    else:
        out_shape = (num_classes, height, width)

    d = Binomial(1, torch.tensor([max_rois / (height * width)]))
    return d.sample(out_shape).squeeze(-1).bool()


@pytest.fixture
def points(batch_size, input_size, num_classes, one_hot_indices, threshold):
    height, width = input_size
    torch.random.manual_seed(42)

    if batch_size is None:
        regs = torch.ones(4, height, width)
        points = torch.zeros(num_classes, height, width)
        positive_points = torch.rand(num_classes, height, width)
    else:
        regs = torch.ones(batch_size, 4, height, width)
        points = torch.zeros(batch_size, num_classes, height, width)
        positive_points = torch.rand(batch_size, num_classes, height, width)

    regs[2:, ...].mul_(2)
    positive_points.clamp_(min=threshold + 0.01)
    _ = torch.where(one_hot_indices, positive_points, points)
    return torch.cat([_, regs], dim=-3)


@pytest.fixture
def bbox(label):
    return label[..., :4]


@pytest.fixture
def classes(label):
    return label[..., 4:]


def test_points_to_anchors(points, upsample, one_hot_indices, max_rois, num_classes, input_size, batch_size, threshold):
    layer = PointsToAnchors(upsample, max_rois, threshold)
    output = layer(points)

    if batch_size is None:
        assert output.shape[0] <= max_rois
        assert output.shape[1] == 6
    else:
        assert output.shape[0] == batch_size
        assert output.shape[1] <= max_rois
        assert output.shape[2] == 6

    cls, score, reg = output[..., :, 4:5].round(), output[..., :, 5:], output[..., :, :4]

    assert ((cls >= -1) & (cls < num_classes)).all()
    assert (reg[..., 0] <= reg[..., 2]).all()
    assert (reg[..., 1] <= reg[..., 3]).all()


@cuda_or_skip
def test_cuda(batch_size):
    layer = PointsToAnchors(2, 10, 0.0)
    if batch_size is not None:
        points = torch.rand(batch_size, 6, 32, 32).cuda()
    else:
        points = torch.rand(6, 32, 32).cuda()
    output = layer(points)
    assert output.device == points.device


def test_input_unchanged():
    layer = PointsToAnchors(2, 10, 0.0)
    points = torch.rand(6, 32, 32)
    points_orig = points.clone()
    layer(points)
    assert torch.allclose(points, points_orig)


@pytest.mark.xfail
def test_corner_case():
    input = torch.tensor(
        [
            [0.0, 1.0, 2.0, 2.0, 200.0, 100.0],
            [0.0, 0.0, -1.0, -1.0, -1.0, -1.0],
            [0.0, 0.0, -1.0, -1.0, -1.0, -1.0],
            [0.0, 0.0, -1.0, -1.0, -1.0, -1.0],
            [1.0, 0.0, 1.0, 1.0, 100.0, 100.0],
            [0.0, 0.0, -1.0, -1.0, -1.0, -1.0],
            [0.0, 0.0, -1.0, -1.0, -1.0, -1.0],
            [0.0, 0.0, -1.0, -1.0, -1.0, -1.0],
            [0.0, 0.0, -1.0, -1.0, -1.0, -1.0],
            [0.0, 0.0, -1.0, -1.0, -1.0, -1.0],
            [0.0, 0.0, -1.0, -1.0, -1.0, -1.0],
            [0.0, 0.0, -1.0, -1.0, -1.0, -1.0],
            [0.0, 0.0, -1.0, -1.0, -1.0, -1.0],
            [0.0, 0.0, -1.0, -1.0, -1.0, -1.0],
            [0.0, 0.0, -1.0, -1.0, -1.0, -1.0],
            [1.0, 0.0, 1.0, 1.0, 100.0, 100.0],
        ]
    )
    input = input.view(2, 8, 6).permute(-1, 0, 1).contiguous()
    layer = PointsToAnchors(1, 10, 0.0)
    output = layer(input)
    expected = torch.tensor(
        [
            [-98.0, -48.0, 102.0, 52.0, 1.0, 1.0],
            [-42.0, -48.0, 58.0, 52.0, 1.0, 0.0],
            [-45.0, -49.0, 55.0, 51.0, 1.0, 0.0],
        ]
    )
    assert torch.allclose(expected, output)
