#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import torch

from combustion.vision import visualize_bbox


@pytest.fixture(
    params=[(1, 32, 32), (32, 32),]
)
def img_shape(request):
    return request.param


@pytest.fixture(params=["Tensor", "np.array"])
def img(request, img_shape):
    img_t = request.param
    tensor = torch.rand(*img_shape)
    if img_t == "Tensor":
        return tensor
    elif img_t == "np.array":
        return tensor.numpy()
    else:
        raise pytest.UsageError(f"unknown type for fixture image: {img_t}")


@pytest.fixture(params=["Tensor", "np.array"])
def label(request):
    label_t = request.param
    tensor = torch.Tensor([[1], [2]])
    if label_t == "Tensor":
        return tensor
    elif label_t == "np.array":
        return tensor.numpy()
    else:
        raise pytest.UsageError(f"unknown type for fixture label: {label_t}")


@pytest.fixture(params=["Tensor", "np.array"])
def bbox(request):
    bbox_t = request.param
    tensor = torch.Tensor([[0, 5, 1, 6], [0, 5, 1, 6]])
    if bbox_t == "Tensor":
        return tensor
    elif bbox_t == "np.array":
        return tensor.numpy()
    else:
        raise pytest.UsageError(f"unknown type for fixture bbox: {bbox_t}")


def test_visualize_bbox(img, label, bbox):
    result = visualize_bbox(img, bbox, label)
    assert result.shape != img.shape or not torch.allclose(torch.as_tensor(img), torch.as_tensor(result))


def test_input_image_unchanged(img, label, bbox):
    # copy original
    if isinstance(img, torch.Tensor):
        original_img = img.clone()
    else:
        original_img = torch.from_numpy(img).clone().numpy()

    visualize_bbox(img, bbox, label)
    assert img.shape == original_img.shape and torch.allclose(torch.as_tensor(img), torch.as_tensor(original_img))


def test_result_channels_first(img, label, bbox):
    result = visualize_bbox(img, bbox, label)
    assert result.shape[-2:] == (32, 32)
