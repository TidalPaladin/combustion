#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from typing import Optional

import pytest
import torch

from combustion.vision import visualize_bbox


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


@pytest.fixture
def label(input_type, img_shape):
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


@pytest.fixture
def scores(input_type, label):
    torch.random.manual_seed(42)
    scores_t = input_type
    tensor = torch.rand_like(torch.Tensor(label))
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

        plt.imsave(path, result.permute(1, 2, 0).cpu().numpy())

    def test_inputs_unchanged(self, img, label, bbox, class_names, scores):
        def copy(x):
            return torch.as_tensor(x).clone()

        img_c = copy(img)
        label_c = copy(label)
        bbox_c = copy(bbox)
        scores_c = copy(scores)

        result = visualize_bbox(img, bbox=bbox, classes=label, scores=scores, class_names=class_names)

        assert torch.allclose(torch.as_tensor(img), img_c)
        assert torch.allclose(torch.as_tensor(label), label_c)
        assert torch.allclose(torch.as_tensor(bbox), bbox_c)
        assert torch.allclose(torch.as_tensor(scores), scores_c)

    def test_visualize_bbox(self, img, label, bbox, class_names, scores):
        if not isinstance(img, torch.Tensor):
            pytest.skip()

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

    def test_class_names(self, img, label, bbox):
        class_names = {1: "foo", 2: "bar"}
        no_names = visualize_bbox(img, bbox, label)
        names = visualize_bbox(img, bbox, label, class_names=class_names)
        assert names.shape == no_names.shape and not torch.allclose(torch.as_tensor(names), torch.as_tensor(no_names))

        if self.DEST is not None and img.ndim == 3:
            dest = os.path.join(self.DEST, "test_class_names.png")
            self.save(dest, names)

    def test_multiple_scores(self, img, label, bbox):
        torch.random.manual_seed(42)
        tensor1 = torch.rand_like(torch.as_tensor(label))
        tensor2 = torch.rand_like(torch.as_tensor(label))
        tensor = torch.cat([tensor1, tensor2], dim=-1)

        scores1 = visualize_bbox(img, bbox, label, scores=tensor1)
        scores2 = visualize_bbox(img, bbox, label, scores=tensor)
        assert scores1.shape == scores2.shape and not torch.allclose(torch.as_tensor(scores1), torch.as_tensor(scores2))

        if self.DEST is not None and img.ndim == 3:
            dest = os.path.join(self.DEST, "test_multiple_scores.png")
            self.save(dest, scores2)
