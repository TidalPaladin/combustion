#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import torch

from combustion.testing import cuda_or_skip
from combustion.vision import AnchorsToPoints, PointsToAnchors
from combustion.vision.centernet import CenterNetMixin


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

    assert offset_x.nonzero(as_tuple=False).numel()
    assert offset_y.nonzero(as_tuple=False).numel()
    assert size_x.nonzero(as_tuple=False).numel()
    assert size_y.nonzero(as_tuple=False).numel()


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


def test_no_box_input():
    image_shape = (64, 64)
    num_classes = 2
    bbox = torch.empty(10, 4).fill_(-1)
    classes = torch.empty(10, 1).fill_(-1)
    downsample = 2

    layer = AnchorsToPoints(num_classes, downsample)
    output = layer(bbox, classes, image_shape)

    height, width = image_shape
    assert output.shape == (num_classes + 4, height // downsample, width // downsample)
    assert (output[0:2] == 0).all()


def test_exception_on_bad_boxes():
    image_shape = (64, 64)
    num_classes = 2
    bbox = torch.tensor([[2.0, 2.0, 2.0, 2.0]])
    classes = torch.tensor([[0.0]])
    downsample = 2

    layer = AnchorsToPoints(num_classes, downsample)
    with pytest.raises(RuntimeError, match="2., 2., 2., 2."):
        layer(bbox, classes, image_shape)


@pytest.mark.parametrize("downsample", [2, 4, 8])
def test_reversible_with_points_to_anchors(downsample):
    image_shape = (16, 16)
    num_classes = 3
    bbox = torch.tensor([[0.0, 00.0, 10.0, 10.0]])
    classes = torch.tensor([[0.0]])

    to_points = AnchorsToPoints(num_classes, downsample)
    to_anchors = PointsToAnchors(downsample, 1)

    midpoint = to_points(bbox, classes, image_shape)
    output = to_anchors(midpoint)

    out_bbox = output[..., :4]
    out_cls = output[..., 6:7]
    out_score = output[..., 5:6]
    assert torch.allclose(out_bbox, bbox)
    assert torch.allclose(out_cls.type_as(classes), classes)


def test_overlapping_boxes():
    image_shape = (16, 16)
    num_classes = 3
    downsample = 8
    bbox = torch.tensor([[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0]])
    classes = torch.tensor([[0.0], [1.0]])

    layer = AnchorsToPoints(num_classes, downsample)
    output = layer(bbox, classes, image_shape)
    assert output[0, 0, 0] == 1.0
    assert output[1, 0, 0] == 1.0
    assert output[0, 1:, 1:] < 1.0
    assert output[1, 1:, 1:] < 1.0


def test_input_unchanged():
    image_shape = (16, 16)
    num_classes = 3
    downsample = 8
    bbox = torch.tensor([[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0]])
    classes = torch.tensor([[0.0], [1.0]])

    bbox_orig = bbox.clone()
    classes_orig = classes.clone()
    layer = AnchorsToPoints(num_classes, downsample)
    layer(bbox, classes, image_shape)

    assert torch.allclose(bbox_orig, bbox)
    assert torch.allclose(classes_orig, classes)


def test_corner_case():
    bbox = torch.tensor(
        [[[115.0, 235.0, 188.0, 345.0], [-1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0]]]
    )
    classes = torch.tensor([[[0.0], [-1.0], [-1.0], [-1.0]]])
    layer = AnchorsToPoints(2, 2, 0.7, 3.0)
    output = layer(bbox, classes, (512, 256))
    assert (output[0, :2, ...] == 1).sum() == 1

    positive_ind = (output[:, :2, ...] == 1).nonzero(as_tuple=False)
    positive_elems = output[0, :, positive_ind[:, -2], positive_ind[:, -1]]
    assert (positive_elems[..., -2:] >= 0.0).all()


class TestCenterNetMixin:
    @pytest.mark.parametrize("split_label", [False, True, [1, 1]])
    def test_split_box_target_result(self, split_label):
        torch.random.manual_seed(42)
        bbox_target = torch.randint(0, 10, (3, 4, 4))
        label_target = torch.randint(0, 10, (3, 4, 2))
        target = torch.cat([bbox_target, label_target], dim=-1)

        mixin = CenterNetMixin()
        result = mixin.split_box_target(target, split_label=split_label)
        bbox = result[0]
        label = result[1:]
        assert torch.allclose(bbox, bbox_target)

        if not split_label:
            assert torch.allclose(torch.cat(label, dim=-1), label_target)
        else:
            for i, sub_label in enumerate(label):
                assert torch.allclose(sub_label, label_target[..., i : i + 1])

    def test_split_box_target_returns_views(self):
        torch.random.manual_seed(42)
        bbox_target = torch.randint(0, 10, (3, 4, 4))
        label_target = torch.randint(0, 10, (3, 4, 2))
        target = torch.cat([bbox_target, label_target], dim=-1)

        mixin = CenterNetMixin()
        bbox, label = mixin.split_box_target(target)

        target.mul_(10)
        assert torch.allclose(bbox, target[..., :4])
        assert torch.allclose(label, target[..., 4:])

    def test_split_point_target(self):
        torch.random.manual_seed(42)
        heatmap_target = torch.rand(3, 2, 10, 10)
        regression_target = torch.rand(3, 4, 10, 10)
        target = torch.cat([heatmap_target, regression_target], dim=-3)

        mixin = CenterNetMixin()
        heatmap, regression = mixin.split_point_target(target)

        assert torch.allclose(heatmap, heatmap_target)
        assert torch.allclose(regression, regression_target)

    def test_split_point_target_returns_views(self):
        torch.random.manual_seed(42)
        heatmap_target = torch.rand(3, 2, 10, 10)
        regression_target = torch.rand(3, 4, 10, 10)
        target = torch.cat([heatmap_target, regression_target], dim=-3)

        mixin = CenterNetMixin()
        heatmap, regression = mixin.split_point_target(target)

        target.mul_(10)
        assert torch.allclose(heatmap, target[..., :-4, :, :])
        assert torch.allclose(regression, target[..., -4:, :, :])

    @pytest.mark.parametrize("extra_labels", [False, True])
    def test_combine_box_target(self, extra_labels):
        torch.random.manual_seed(42)
        bbox_target = torch.randint(0, 10, (3, 4, 4))
        label_target = torch.randint(0, 10, (3, 4, 2))
        if extra_labels:
            extra_labels = (torch.randint(0, 10, (3, 4, 2)),)
        else:
            extra_labels = ()

        true_target = torch.cat([bbox_target, label_target, *extra_labels], dim=-1)

        mixin = CenterNetMixin()
        target = mixin.combine_box_target(bbox_target, label_target, *extra_labels)

        assert torch.allclose(target, true_target)

    def test_combine_point_target(self):
        torch.random.manual_seed(42)
        heatmap_target = torch.rand(3, 2, 10, 10)
        regression_target = torch.rand(3, 4, 10, 10)
        true_target = torch.cat([heatmap_target, regression_target], dim=-3)

        mixin = CenterNetMixin()
        target = mixin.combine_point_target(heatmap_target, regression_target)

        assert torch.allclose(target, true_target)

    def test_heatmap_max_score(self):
        torch.random.manual_seed(42)
        heatmap = torch.rand(3, 2, 10, 10)

        expected = heatmap.max(dim=-1).values.max(dim=-1).values

        mixin = CenterNetMixin()
        actual = mixin.heatmap_max_score(heatmap)

        assert actual.ndim == 2
        assert torch.allclose(actual, expected)

    def test_visualize_heatmap_no_background(self):
        torch.random.manual_seed(42)
        heatmap = torch.rand(3, 2, 10, 10)

        mixin = CenterNetMixin()
        result = mixin.visualize_heatmap(heatmap)

        assert len(result) == heatmap.shape[-3]

        for x in result:
            assert x.min() >= 0
            assert x.max() <= 255
            assert x.dtype == torch.uint8

    def test_visualize_heatmap_background(self):
        torch.random.manual_seed(42)
        heatmap = torch.rand(3, 2, 10, 10)
        background = torch.rand(3, 1, 10, 10)

        mixin = CenterNetMixin()
        result = mixin.visualize_heatmap(heatmap, background=background)

        assert len(result) == heatmap.shape[-3]

        for x in result:
            assert x.min() >= 0
            assert x.max() <= 255
            assert x.dtype == torch.uint8

    @pytest.mark.parametrize("pad_value", [-1, -2])
    def test_batch_box_target(self, pad_value):
        torch.random.manual_seed(42)
        target1 = torch.randint(0, 10, (3, 6))
        target2 = torch.randint(0, 10, (2, 6))

        mixin = CenterNetMixin()
        batch = mixin.batch_box_target([target1, target2], pad_value=pad_value)

        assert batch.shape[0] == 2
        assert torch.allclose(batch[0], target1)
        assert torch.allclose(batch[1, :2, :], target2)
        assert (batch[1, 2, :] == pad_value).all()

    @pytest.mark.parametrize("pad_value", [-1, -2])
    def test_unbatch_box_target(self, pad_value):
        torch.random.manual_seed(42)
        target = torch.randint(0, 10, (2, 3, 6))
        target[0, 2, :].fill_(pad_value)

        mixin = CenterNetMixin()
        split_batch = mixin.unbatch_box_target(target, pad_value=pad_value)

        assert torch.allclose(target[0, :2, ...], split_batch[0])
        assert torch.allclose(target[1], split_batch[1])

    @pytest.mark.parametrize("pad_value", [-1, -2])
    def test_flatten_box_target(self, pad_value):
        torch.random.manual_seed(42)
        target = torch.randint(0, 10, (2, 3, 6))
        target[0, 2, :].fill_(pad_value)

        mixin = CenterNetMixin()
        flat_batch = mixin.flatten_box_target(target, pad_value=pad_value)

        assert flat_batch.ndim == 2
        assert flat_batch.shape[0] == 2 + 3

        expected = torch.cat([target[0, :2, ...], target[1, ...]], dim=0)
        assert torch.allclose(flat_batch, expected)
