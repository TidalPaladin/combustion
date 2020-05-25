#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import torch

from combustion.testing import cuda_or_skip
from combustion.vision import AnchorsToPoints


@pytest.fixture(params=[None, 1, 2])
def batch_size(request):
    return request.param


@pytest.fixture(params=[1, 10])
def num_rois(request):
    return request.param


@pytest.fixture(params=[(32, 32), (128, 64)])
def image_size(request):
    return request.param


@pytest.fixture(params=[2, 4])
def num_classes(request):
    return request.param


@pytest.fixture
def label(batch_size, num_rois, image_size, num_classes, cuda):
    img_h, img_w = image_size
    torch.random.manual_seed(42)

    if batch_size is None:
        label = torch.empty(num_rois, 5)
    else:
        label = torch.empty(batch_size, num_rois, 5)

    label[..., 0] = torch.randint(0, img_w // 2, label[..., 0].shape)
    label[..., 1] = torch.randint(0, img_h // 2, label[..., 1].shape)
    label[..., 2] = torch.randint(img_w // 2, img_w, label[..., 2].shape)
    label[..., 3] = torch.randint(img_h // 2, img_h, label[..., 3].shape)

    label[..., 4] = torch.randint(0, num_classes, label[..., 4].shape)
    if cuda:
        label = label.cuda()
    return label


@pytest.fixture
def bbox(label):
    return label[..., :4]


@pytest.fixture
def classes(label):
    return label[..., 4:]


@pytest.fixture(params=[1, 3])
def image_shape(request, batch_size, image_size):
    return (batch_size, request.param, *image_size)


def test_anchors_to_points(bbox, classes, num_classes, image_shape, batch_size):
    height, width = image_shape[-2:]
    downsample = 2
    layer = AnchorsToPoints(num_classes, downsample)
    output = layer(bbox, classes, image_shape)

    if batch_size is not None:
        assert output.shape == (batch_size, num_classes + 4, height // downsample, width // downsample)
    else:
        assert output.shape == (num_classes + 4, height // downsample, width // downsample)

    cls = output[..., :num_classes, :, :]
    reg = output[..., num_classes:, :, :]
    assert cls.min() >= 0.0
    assert cls.max() <= 1.0

    offset_x, offset_y = reg[..., 0, :, :], reg[..., 1, :, :]
    size_x, size_y = reg[..., 2, :, :], reg[..., 3, :, :]

    assert offset_x.nonzero().numel()
    assert offset_y.nonzero().numel()
    assert size_x.nonzero().numel()
    assert size_y.nonzero().numel()


@cuda_or_skip
def test_cuda():
    torch.random.manual_seed(42)
    img_h, img_w = 64, 64
    num_classes = 3

    label = torch.empty(10, 5)
    label[..., 0] = torch.randint(0, img_w // 2, label[..., 0].shape)
    label[..., 1] = torch.randint(0, img_h // 2, label[..., 1].shape)
    label[..., 2] = torch.randint(img_w // 2, img_w, label[..., 2].shape)
    label[..., 3] = torch.randint(img_h // 2, img_h, label[..., 3].shape)
    label[..., 4] = torch.randint(0, num_classes, label[..., 4].shape)
    label = label.cuda()
    bbox, classes = label[..., :4], label[..., 4:]

    layer = AnchorsToPoints(num_classes, 2)
    output = layer(bbox, classes, (img_h, img_w))
    assert output.device == bbox.device
