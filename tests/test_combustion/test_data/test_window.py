#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import torch

from combustion.data.window import DenseWindow, SparseWindow


DEPTH_DIM = 1


@pytest.fixture(params=["sparse", "dense"])
def window_type(request):
    if request.param == "sparse":
        return SparseWindow
    elif request.param == "dense":
        return SparseWindow
    else:
        raise pytest.UsageError(f"unknown window {request.param}")


@pytest.mark.parametrize(
    "window_type,before,after,exp",
    [
        pytest.param(DenseWindow, 1, 0, 2, id="dense_1->0"),
        pytest.param(DenseWindow, 0, 1, 2, id="dense_0->1"),
        pytest.param(DenseWindow, 2, 2, 5, id="dense_2->2"),
        pytest.param(SparseWindow, 1, 0, 2, id="sparse_1->0"),
        pytest.param(SparseWindow, 0, 1, 2, id="sparse_0->1"),
        pytest.param(SparseWindow, 2, 2, 5, id="sparse_2->2"),
    ],
)
def test_length(window_type, before, after, exp):
    window = window_type(before, after)
    assert len(window) == exp


@pytest.mark.parametrize(
    "window_type,before,after,num,exp",
    [
        pytest.param(DenseWindow, 1, 0, 11, 10, id="dense_1->0"),
        pytest.param(DenseWindow, 0, 1, 12, 11, id="dense_0->1"),
        pytest.param(DenseWindow, 2, 2, 10, 6, id="dense_2->2"),
        pytest.param(SparseWindow, 1, 0, 7, 6, id="sparse_1->0"),
        pytest.param(SparseWindow, 0, 1, 5, 4, id="sparse_0->1"),
        pytest.param(SparseWindow, 2, 2, 5, 1, id="sparse_2->2"),
    ],
)
def test_estimate_size(window_type, before, after, num, exp):
    window = window_type(before, after)
    assert window.estimate_size(num) == exp


@pytest.mark.parametrize(
    "window_type,before,after,num",
    [
        pytest.param(DenseWindow, 1, 0, 11, id="dense_1->0"),
        pytest.param(DenseWindow, 0, 1, 12, id="dense_0->1"),
        pytest.param(DenseWindow, 2, 2, 10, id="dense_2->2"),
        pytest.param(SparseWindow, 1, 0, 7, id="sparse_1->0"),
        pytest.param(SparseWindow, 0, 1, 5, id="sparse_0->1"),
        pytest.param(SparseWindow, 2, 2, 5, id="sparse_2->2"),
    ],
)
def test_call(window_type, before, after, num):
    window = window_type(before, after)
    examples = [(torch.ones(2, 2),) * 2] * num
    for x, y in window(examples):
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)

        if window_type == DenseWindow:
            frame_shape = (1, before + after + 1, 2, 2)
        elif before and after:
            frame_shape = (1, 3, 2, 2)
        else:
            frame_shape = (1, 2, 2, 2)
        label_shape = (1, 2, 2)
        assert x.shape == frame_shape
        assert y.shape == label_shape
    assert len(list(window(examples))) == num - before - after
