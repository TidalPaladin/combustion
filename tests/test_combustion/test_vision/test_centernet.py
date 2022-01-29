#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import torch

from combustion.testing import cuda_or_skip
from combustion.vision import AnchorsToPoints, PointsToAnchors
from combustion.vision.centernet import CenterNetMixin


pytestmark = pytest.mark.skip()


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

    out_bbox: Tensor = output[..., :4]  # type: ignore
    out_cls: Tensor = output[..., 6:7]  # type: ignore
    out_score: Tensor = output[..., 5:6]  # type: ignore
    assert torch.allclose(out_bbox, bbox)
    assert torch.allclose(out_cls.type_as(classes), classes)


@pytest.mark.parametrize("max_roi", [1, 2, None])
def test_points_to_anchors_max_roi(max_roi):
    torch.random.manual_seed(42)
    image_shape = (32, 32)
    num_classes = 3
    heatmap = torch.rand(3, num_classes + 4, *image_shape)

    to_anchors = PointsToAnchors(2, max_roi=max_roi)
    result: Tensor = to_anchors(heatmap)  # type: ignore

    if max_roi is not None:
        assert result.shape[-2] <= max_roi


# TODO this test should be more thorough


@pytest.mark.parametrize("max_roi", [1, 2, None])
def test_points_to_anchors_return_indices(max_roi):
    torch.random.manual_seed(42)
    image_shape = (32, 32)
    num_classes = 3
    heatmap = torch.rand(3, num_classes + 4, *image_shape)

    to_anchors = PointsToAnchors(2, max_roi=max_roi)
    result, indices = to_anchors(heatmap, return_indices=True)
    assert indices.shape[:-1] == result.shape[:-1]


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


@pytest.mark.skip(reason="kornia change")
class TestCenterNetMixin:
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
        num_classes = 3
        heatmap = torch.rand(3, num_classes + 4, 10, 10)
        expected = heatmap[..., :num_classes, :, :].max(dim=-1).values.max(dim=-1).values

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

    @pytest.mark.parametrize("label_size", [1, 2])
    def test_append_heatmap_label(self, label_size):
        torch.random.manual_seed(42)
        old_label = torch.rand(3, 6, 10, 10)
        new_label = torch.rand(3, label_size, 10, 10)

        mixin = CenterNetMixin()
        final_label = mixin.append_heatmap_label(old_label, new_label)
        assert torch.allclose(final_label[..., :2, :, :], old_label[..., :2, :, :])
        assert torch.allclose(final_label[..., -4:, :, :], old_label[..., -4:, :, :])
        assert torch.allclose(final_label[..., 2:-4, :, :], new_label)

    @pytest.mark.parametrize("batched", [False, True])
    def test_get_global_pred_target_pairs(self, batched):
        torch.random.manual_seed(42)
        num_classes = 3

        pred_heatmap = torch.rand(num_classes + 4, 10, 10)

        # classes {0, 1} present, class 3 not present
        target = torch.tensor(
            [
                [0, 0, 2.1, 2.1, 0],
                [0, 0, 2, 2, 0],
                [3, 3, 6, 6, 0],
                [5, 5, 10, 9, 1],
                [1, 1, 4.9, 4.9, 1],
                [-1, -1, -1, -1, -1],
            ]
        ).float()

        if batched:
            pred_heatmap = pred_heatmap.unsqueeze_(0).expand(2, -1, -1, -1)
            target = target.unsqueeze_(0).expand(2, -1, -1)

        mixin = CenterNetMixin()
        result = mixin.get_global_pred_target_pairs(pred_heatmap, target)

        # expected pred is the max over the heatmap
        expected_pred = pred_heatmap[..., :-4, :, :].max(dim=-1).values.max(dim=-1).values

        # expected target is 1, 1, 0 for classes 0, 1 present / 2 not present
        expected_target = torch.tensor([1.0, 1.0, 0.0])
        if batched:
            expected_target = expected_target.unsqueeze_(0).expand(2, -1)

        assert torch.allclose(result[..., 0], expected_pred)
        assert torch.allclose(result[..., 1], expected_target)

    def test_split_regression(self):
        torch.random.manual_seed(42)
        regression = torch.rand(3, 4, 10, 10)

        mixin = CenterNetMixin()
        offset, size = mixin.split_regression(regression)

        assert torch.allclose(offset, regression[..., :2, :, :])
        assert torch.allclose(size, regression[..., 2:, :, :])

    def test_combine_regression(self):
        torch.random.manual_seed(42)
        regression = torch.rand(3, 4, 10, 10)

        mixin = CenterNetMixin()
        offset = regression[..., :2, :, :]
        size = regression[..., 2:, :, :]

        result = mixin.combine_regression(offset, size)
        assert torch.allclose(result, regression)

    @pytest.mark.parametrize("with_regression", [False, True])
    @pytest.mark.parametrize("return_inverse", [False, True])
    @pytest.mark.parametrize("keep_classes", [[0], [0, 1], [0, 2]])
    def test_filter_heatmap_classes(self, return_inverse, keep_classes, with_regression):
        torch.random.manual_seed(42)
        mixin = CenterNetMixin()

        possible_classes = set([0, 1, 2])
        if return_inverse:
            drop_classes = set(keep_classes)
            real_keep_classes = possible_classes - drop_classes
        else:
            drop_classes = possible_classes - set(keep_classes)
            real_keep_classes = keep_classes

        if with_regression:
            heatmap = torch.rand(2, len(possible_classes) + 4, 32, 32)
        else:
            heatmap = torch.rand(2, len(possible_classes), 32, 32)

        result = mixin.filter_heatmap_classes(
            heatmap, keep_classes=keep_classes, return_inverse=return_inverse, with_regression=with_regression
        )

        assert result.shape[-3] == heatmap.shape[-3] - len(drop_classes)
        assert result.shape[-2:] == heatmap.shape[-2:]
        assert result.shape[0] == heatmap.shape[0]

        if with_regression:
            expected = heatmap[..., tuple(real_keep_classes) + (-4, -3, -2, -1), :, :]
        else:
            expected = heatmap[..., tuple(real_keep_classes), :, :]

        assert torch.allclose(result, expected)
