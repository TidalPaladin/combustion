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


@pytest.mark.parametrize("max_roi", [1, 2, None])
def test_points_to_anchors_max_roi(max_roi):
    torch.random.manual_seed(42)
    image_shape = (32, 32)
    num_classes = 3
    heatmap = torch.rand(3, num_classes + 4, *image_shape)

    to_anchors = PointsToAnchors(2, max_roi=max_roi)
    result = to_anchors(heatmap)

    if max_roi is not None:
        assert result.shape[-2] <= max_roi
    else:
        assert result.shape[-2] == 372


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

    @pytest.mark.parametrize("label_size", [1, 2])
    def test_append_bbox_label(self, label_size):
        torch.random.manual_seed(42)
        old_label = torch.randint(0, 10, (2, 4, 6))
        new_label = torch.randint(0, 10, (2, 4, label_size))

        mixin = CenterNetMixin()
        final_label = mixin.append_bbox_label(old_label, new_label)
        assert torch.allclose(final_label, torch.cat([old_label, new_label], dim=-1))

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

    @pytest.mark.parametrize("split_scores", [False, True, [1, 1]])
    def test_split_bbox_scores_class_result(self, split_scores):
        torch.random.manual_seed(42)
        bbox_target = torch.randint(0, 10, (3, 4, 4)).float()
        scores_target = torch.rand(3, 4, 2)
        cls_target = torch.randint(0, 10, (3, 4, 1)).float()
        target = torch.cat([bbox_target, scores_target, cls_target], dim=-1)

        mixin = CenterNetMixin()
        result = mixin.split_bbox_scores_class(target, split_scores=split_scores)
        bbox = result[0]
        scores = result[1:-1]
        cls = result[-1]

        assert torch.allclose(bbox, bbox_target)
        assert torch.allclose(cls, cls_target)

        if not split_scores:
            assert torch.allclose(torch.cat(scores, dim=-1), scores_target)
        else:
            for i, sub_scores in enumerate(scores):
                assert torch.allclose(sub_scores, scores_target[..., i : i + 1])

    def test_split_bbox_scores_class_returns_views(self):
        torch.random.manual_seed(42)
        bbox_target = torch.randint(0, 10, (3, 4, 4)).float()
        scores_target = torch.rand(3, 4, 2)
        torch.randint(0, 10, (3, 4, 1)).float()
        target = torch.cat([bbox_target, scores_target], dim=-1)

        mixin = CenterNetMixin()
        result = mixin.split_bbox_scores_class(target)
        bbox = result[0]
        scores = result[1]
        cls = result[-1]

        target.mul_(10)
        assert torch.allclose(bbox, target[..., :4])
        assert torch.allclose(scores, target[..., 4:-1])
        assert torch.allclose(cls, target[..., -1:])

    @pytest.mark.parametrize("extra_scores", [False, True])
    def test_combine_bbox_scores_cls(self, extra_scores):
        torch.random.manual_seed(42)
        bbox = torch.randint(0, 10, (3, 4, 4)).float()
        scores = torch.rand(3, 4, 2)
        cls = torch.randint(0, 10, (3, 4, 1))
        if extra_scores:
            extra_scores = (torch.rand(3, 4, 2).float(),)
        else:
            extra_scores = ()

        true_target = torch.cat([bbox, scores, *extra_scores, cls], dim=-1)

        mixin = CenterNetMixin()
        target = mixin.combine_bbox_scores_class(bbox, cls, scores, *extra_scores)

        assert torch.allclose(target, true_target)

    @pytest.mark.parametrize("true_positive_limit", [True, False])
    def test_get_pred_target_pairs(self, true_positive_limit):
        torch.random.manual_seed(42)
        num_classes = 3

        # its easier to check correctness with bounding boxes than heatmaps,
        # so we start with bounding boxes and compute an equaivalent heatmap
        #
        # assume we predicted a set of boxes (x1, y1, x2, y2, class)
        pred_bbox = torch.tensor(
            [
                [0, 0, 2, 2, 0],  # overlaps target[0] and target[1]
                [2, 2, 4, 4, 0],  # false positive
                [1, 1, 4, 4, 1],  # overlaps target[4] better than V
                [2, 2, 6, 6, 1],  # overlaps target[4] worse than ^
                [5, 5, 9, 9, 1],  # overlaps target[2]
            ]
        ).float()

        # compute the equivalent heatmap
        atp = AnchorsToPoints(num_classes=num_classes, downsample=1)
        pred_heatmap = atp(pred_bbox[..., :4], pred_bbox[..., -1:], shape=(10, 10))

        # for clarity make max heatmap probability just under 1
        pred_heatmap[..., :num_classes, :, :].clamp_max_(0.99)
        assert (pred_heatmap == 0.99).sum() == pred_bbox.shape[-2], "heatmap discretization dropped a box"

        target = torch.tensor(
            [
                [0, 0, 2.1, 2.1, 0],  # tp pred_bbox[0]
                [0, 0, 2, 2, 0],  # tp pred_bbox[0]
                [3, 3, 6, 6, 0],  # false negative
                [5, 5, 10, 9, 1],  # tp pred_bbox[4]
                [1, 1, 4.9, 4.9, 1],  # tp pred_bbox[3]
                [-1, -1, -1, -1, -1],  # padding
            ]
        ).float()

        mixin = CenterNetMixin()
        pred_score, target_class, is_correct = mixin.get_pred_target_pairs(
            pred_heatmap, target, upsample=1, true_positive_limit=true_positive_limit, iou_threshold=0.3
        )

        assert torch.allclose(pred_score, torch.tensor([0.99, 0.99, 0.99, 0.99, 0.99, 0.0]))

        tp = is_correct.sum()
        fp = ((~is_correct) & (pred_score > 0)).sum()
        fn = (pred_score == 0).sum()

        if true_positive_limit:
            assert tp == 3
            assert fp == 2
            assert fn == 1
        else:
            assert tp == 4  # one extra tp from duplicate boxes
            assert fp == 1  # one less fp from ^
            assert fn == 1

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
