#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from typing import Optional

import pytest
import torch

from combustion.vision import (
    append_bbox_label,
    batch_box_target,
    combine_bbox_scores_class,
    combine_box_target,
    filter_bbox_classes,
    flatten_box_target,
    split_bbox_scores_class,
    split_box_target,
    unbatch_box_target,
    visualize_bbox,
)


@pytest.fixture(
    params=[
        (1, 227, 227),
        (3, 227, 227),
        (227, 227),
        (3, 1, 227, 227),
        (1, 1, 227, 227),
    ]
)
def img_shape(request):
    return request.param


@pytest.fixture(params=["Tensor", "np.array"])
def input_type(request):
    return request.param


@pytest.fixture
def img(input_type, img_shape):
    torch.random.manual_seed(42)
    img_t = input_type
    tensor = torch.rand(*img_shape)
    if img_t == "Tensor":
        return tensor
    elif img_t == "np.array":
        return tensor.numpy()
    else:
        raise pytest.UsageError(f"unknown type for fixture image: {img_t}")


@pytest.fixture(params=["classes", "no_classes"])
def label(request, input_type, img_shape):
    if request.param == "no_classes":
        return None
    label_t = input_type
    tensor = torch.Tensor([[1], [2]])
    if len(img_shape) == 4:
        tensor = tensor.unsqueeze_(0).repeat(img_shape[0], 1, 1)
    if label_t == "Tensor":
        return tensor
    elif label_t == "np.array":
        return tensor.numpy()
    else:
        raise pytest.UsageError(f"unknown type for fixture label: {label_t}")


@pytest.fixture
def bbox(input_type, img_shape):
    bbox_t = input_type
    tensor = torch.Tensor([[20, 20, 100, 100], [60, 60, 100, 100]])
    if len(img_shape) == 4:
        tensor = tensor.unsqueeze_(0).repeat(img_shape[0], 1, 1)
    if bbox_t == "Tensor":
        return tensor
    elif bbox_t == "np.array":
        return tensor.numpy()
    else:
        raise pytest.UsageError(f"unknown type for fixture bbox: {bbox_t}")


@pytest.fixture
def class_names():
    return {x: str(x) for x in range(5)}


@pytest.fixture(params=["scores", "no_scores"])
def scores(request, input_type, img_shape):
    if request.param == "no_scores":
        return None
    torch.random.manual_seed(42)
    scores_t = input_type
    if len(img_shape) == 4:
        tensor = torch.rand(img_shape[0], 2, 1)
    else:
        tensor = torch.rand(2, 1)
    if scores_t == "Tensor":
        return tensor
    elif scores_t == "np.array":
        return tensor.numpy()
    else:
        raise pytest.UsageError(f"unknown type for fixture bbox: {bbox_t}")


class TestVisualizeBbox:

    # Set this to a directory to write out some sample images from test cases
    # DEST: Optional[str] = None
    DEST: Optional[str] = "/home/tidal"

    def save(self, path, result):
        import matplotlib.pyplot as plt

        if os.path.isdir(path):
            plt.imsave(path, result.permute(1, 2, 0).cpu().numpy())

    def test_inputs_unchanged(self, img, label, bbox, class_names, scores):
        def copy(x):
            return torch.as_tensor(x).clone()

        img_c = copy(img)
        bbox_c = copy(bbox)
        if label is not None:
            label_c = copy(label)
        if scores is not None:
            scores_c = copy(scores)

        result = visualize_bbox(img, bbox=bbox, classes=label, scores=scores, class_names=class_names)

        assert torch.allclose(torch.as_tensor(img), img_c)
        assert torch.allclose(torch.as_tensor(bbox), bbox_c)
        if label is not None:
            assert torch.allclose(torch.as_tensor(label), label_c)
        if scores is not None:
            assert torch.allclose(torch.as_tensor(scores), scores_c)

    def test_visualize_bbox(self, img, label, bbox, class_names, scores):
        if not isinstance(img, torch.Tensor):
            pytest.skip()

        result = visualize_bbox(img, bbox, label, scores, class_names)
        assert isinstance(result, torch.Tensor)
        assert result.shape[-2:] == img.shape[-2:]
        assert result.shape[-3] == 3
        if img.ndim != 2:
            assert result.ndim == img.ndim

        if self.DEST is not None and img.ndim == 3:
            dest = os.path.join(self.DEST, "test_visualize_bbox.png")
            self.save(dest, result)

    def test_class_names(self, img, label, bbox):
        if label is None:
            pytest.skip()
        class_names = {1: "foo", 2: "bar"}
        no_names = visualize_bbox(img, bbox, label)
        names = visualize_bbox(img, bbox, label, class_names=class_names)
        assert names.shape == no_names.shape and not torch.allclose(torch.as_tensor(names), torch.as_tensor(no_names))

        if self.DEST is not None and img.ndim == 3:
            dest = os.path.join(self.DEST, "test_class_names.png")
            self.save(dest, names)

    def test_multiple_scores(self, img, label, bbox, scores):
        torch.random.manual_seed(42)
        if scores is None:
            pytest.skip()
        tensor1 = torch.rand_like(torch.as_tensor(scores))
        tensor2 = torch.rand_like(torch.as_tensor(scores))
        tensor = torch.cat([tensor1, tensor2], dim=-1)

        scores1 = visualize_bbox(img, bbox, label, scores=tensor1)
        scores2 = visualize_bbox(img, bbox, label, scores=tensor)
        assert scores1.shape == scores2.shape and not torch.allclose(torch.as_tensor(scores1), torch.as_tensor(scores2))

        if self.DEST is not None and img.ndim == 3:
            dest = os.path.join(self.DEST, "test_multiple_scores.png")
            self.save(dest, scores2)

    @pytest.mark.usefixtures("cuda_or_skip")
    def test_cuda(self, img, label, bbox, class_names, scores):
        if not isinstance(img, torch.Tensor):
            pytest.skip()

        img = img.cuda()
        bbox = bbox.cuda()
        label = label.cuda() if label is not None else None
        scores = scores.cuda() if scores is not None else None
        result = visualize_bbox(img, bbox, label, scores, class_names)
        assert isinstance(result, torch.Tensor)
        assert result.dtype == torch.uint8
        assert result.shape[-2:] == img.shape[-2:]
        assert result.shape[-3] == 3
        if img.ndim != 2:
            assert result.ndim == img.ndim

        if self.DEST is not None and img.ndim == 3:
            dest = os.path.join(self.DEST, "test_visualize_bbox.png")
            self.save(dest, result)


class TestBboxHelpers:
    @pytest.mark.parametrize("split_label", [False, True, [1, 1]])
    def test_split_box_target_result(self, split_label):
        torch.random.manual_seed(42)
        bbox_target = torch.randint(0, 10, (3, 4, 4))
        label_target = torch.randint(0, 10, (3, 4, 2))
        target = torch.cat([bbox_target, label_target], dim=-1)

        result = split_box_target(target, split_label=split_label)
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

        bbox, label = split_box_target(target)

        target.mul_(10)
        assert torch.allclose(bbox, target[..., :4])
        assert torch.allclose(label, target[..., 4:])

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

        target = combine_box_target(bbox_target, label_target, *extra_labels)

        assert torch.allclose(target, true_target)

    @pytest.mark.parametrize("pad_value", [-1, -2])
    def test_batch_box_target(self, pad_value):
        torch.random.manual_seed(42)
        target1 = torch.randint(0, 10, (3, 6))
        target2 = torch.randint(0, 10, (2, 6))

        batch = batch_box_target([target1, target2], pad_value=pad_value)

        assert batch.shape[0] == 2
        assert torch.allclose(batch[0], target1)
        assert torch.allclose(batch[1, :2, :], target2)
        assert (batch[1, 2, :] == pad_value).all()

    def test_batch_box_target_batched_inputs(self):
        torch.random.manual_seed(42)
        target1 = torch.randint(0, 10, (3, 3, 6))
        target2 = torch.randint(0, 10, (2, 2, 6))

        batch = batch_box_target([target1, target2])
        assert batch.shape[-1] == 6
        assert batch.shape[-2] == 3
        assert batch.shape[0] == 5  # 3 + 2
        assert (batch[-1, -1, :] == -1).all()

    @pytest.mark.parametrize("pad_value", [-1, -2])
    def test_unbatch_box_target(self, pad_value):
        torch.random.manual_seed(42)
        target = torch.randint(0, 10, (2, 3, 6))
        target[0, 2, :].fill_(pad_value)

        split_batch = unbatch_box_target(target, pad_value=pad_value)

        assert torch.allclose(target[0, :2, ...], split_batch[0])
        assert torch.allclose(target[1], split_batch[1])

    @pytest.mark.parametrize("pad_value", [-1, -2])
    def test_flatten_box_target(self, pad_value):
        torch.random.manual_seed(42)
        target = torch.randint(0, 10, (2, 3, 6))
        target[0, 2, :].fill_(pad_value)

        flat_batch = flatten_box_target(target, pad_value=pad_value)

        assert flat_batch.ndim == 2
        assert flat_batch.shape[0] == 2 + 3

        expected = torch.cat([target[0, :2, ...], target[1, ...]], dim=0)
        assert torch.allclose(flat_batch, expected)

    @pytest.mark.parametrize("label_size", [1, 2])
    def test_append_bbox_label(self, label_size):
        torch.random.manual_seed(42)
        old_label = torch.randint(0, 10, (2, 4, 6))
        new_label = torch.randint(0, 10, (2, 4, label_size))

        final_label = append_bbox_label(old_label, new_label)
        assert torch.allclose(final_label, torch.cat([old_label, new_label], dim=-1))

    @pytest.mark.parametrize("split_scores", [False, True, [1, 1]])
    def test_split_bbox_scores_class_result(self, split_scores):
        torch.random.manual_seed(42)
        bbox_target = torch.randint(0, 10, (3, 4, 4)).float()
        scores_target = torch.rand(3, 4, 2)
        cls_target = torch.randint(0, 10, (3, 4, 1)).float()
        target = torch.cat([bbox_target, scores_target, cls_target], dim=-1)

        result = split_bbox_scores_class(target, split_scores=split_scores)
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

        result = split_bbox_scores_class(target)
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

        target = combine_bbox_scores_class(bbox, cls, scores, *extra_scores)

        assert torch.allclose(target, true_target)

    @pytest.mark.parametrize("return_inverse", [False, True])
    @pytest.mark.parametrize("pad_value", [-1, -2])
    @pytest.mark.parametrize("keep_classes", [[0], [0, 1], [0, 2]])
    def test_filter_bbox_classes(self, return_inverse, pad_value, keep_classes):
        torch.random.manual_seed(42)

        possible_classes = set([0, 1, 2])
        if return_inverse:
            set(keep_classes)
        else:
            possible_classes - set(keep_classes)

        classes1 = torch.tensor([0, 1, 2, 1, 0, 0, -1, -1, -1]).unsqueeze_(-1)
        classes2 = torch.tensor([1, 0, 0, 1, 0, -1, -1, -1, -1]).unsqueeze_(-1)

        target = torch.stack([classes1, classes2], dim=0)
        bbox = torch.randint(0, 10, (2, target.shape[-2], 4))
        target = torch.cat([bbox, target], dim=-1).float()

        result = filter_bbox_classes(target, keep_classes=keep_classes, return_inverse=return_inverse)

        for cls in possible_classes:
            if return_inverse and cls in keep_classes:
                assert not (result[..., -1] == cls).any()
            if not return_inverse and cls not in keep_classes:
                assert not (result[..., -1] == cls).any()
