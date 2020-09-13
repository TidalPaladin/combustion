#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import torch

from combustion.testing import cuda_or_skip
from combustion.vision.anchors import AnchorBoxTransform, Anchors, ClipBoxes


@pytest.fixture(params=[(32, 32), (64, 64), (32, 16)])
def image(request):
    height, width = request.param
    return torch.rand(1, 3, height, width)


@pytest.fixture(
    params=[
        pytest.param(
            [
                1,
            ],
            id="levels=1",
        ),
        pytest.param([2, 3], id="levels=2,3"),
    ]
)
def levels(request):
    return request.param


def test_generate_anchors(image, levels):
    layer = Anchors(levels)
    len(layer.ratios) * len(layer.scales)
    anchors = layer(image)
    assert anchors.shape[-1] == 4


def test_num_anchors_genreated():
    levels = [
        0,
    ]
    image = torch.rand(1, 3, 4, 4)
    sizes = [
        2,
    ]
    strides = [
        1,
    ]

    layer = Anchors(levels, sizes=sizes, strides=strides)
    anchors = layer(image)
    assert anchors.shape[-1] == 4


def test_clip_boxes(image):
    boxes = (
        torch.tensor(
            [
                [0, 10, 10, 20],
                [-10, 10, 20, image.shape[-1] + 1],
                [0, 10, image.shape[-2] + 1, 20],
            ]
        )
        .unsqueeze(0)
        .type_as(image)
    )

    layer = ClipBoxes()
    output = layer(boxes, image)
    assert (output >= 0).all()
    assert (output[:, :, 2] <= image.shape[-1]).all()
    assert (output[:, :, 3] <= image.shape[-2]).all()


@pytest.mark.parametrize("log_length", [True, False])
@pytest.mark.parametrize("mean", [None, torch.tensor([0.1, 0.1, 0.2, 0.2])])
@pytest.mark.parametrize("std", [None, torch.tensor([0.1, 0.1, 0.2, 0.2])])
def test_anchor_box_transform(image, log_length, mean, std):
    boxes = torch.tensor([[0, 0, 10, 10]]).unsqueeze(0).type_as(image)
    deltas = torch.tensor([[0, 0, 1, 1]]).unsqueeze(0).type_as(image)

    if log_length:
        deltas[..., -2:].log_()
    if mean is not None:
        deltas.sub_(mean)
    if std is not None:
        deltas.div_(std)

    layer = AnchorBoxTransform(mean=mean, std=std, log_length=log_length)
    output = layer(boxes, deltas)
    assert torch.allclose(boxes, output)


@cuda_or_skip
def test_memory(levels):
    image = torch.rand(1, 3, 64, 64)
    layer = Anchors(levels)
    image = image.cuda()
    len(layer.ratios) * len(layer.scales)

    torch.cuda.reset_peak_memory_stats()
    memory_before = torch.cuda.max_memory_allocated()
    anchors = layer(image)
    memory_after = torch.cuda.max_memory_allocated()
    memory_delta = memory_after - memory_before

    num_anchors = 9
    bytes_per_float = 4
    coords_per_box = 4
    num_levels = len(levels)

    assert memory_delta <= anchors.numel() * num_anchors * bytes_per_float * coords_per_box * num_levels * 4
