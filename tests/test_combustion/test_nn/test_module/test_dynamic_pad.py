#!/usr/bin/env python
# -*- coding: utf-8 -*-

from math import ceil

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from combustion.nn import DynamicSamePad, MatchShapes
from combustion.nn.functional import patch_dynamic_same_pad
from combustion.testing import TorchScriptTestMixin


@pytest.fixture(params=[1, 3, 5])
def kernel_size(request):
    return request.param


@pytest.fixture(params=[1, 2, 3])
def stride(request):
    return request.param


@pytest.fixture(
    params=[
        pytest.param("constant"),
        pytest.param("reflect"),
        pytest.param("replicate"),
        pytest.param("circular"),
    ]
)
def padding_mode(request):
    return request.param


@pytest.fixture(
    params=[
        pytest.param(lambda k, s: nn.Conv1d(1, 1, k, stride=s), id="conv1d"),
        pytest.param(lambda k, s: nn.Conv3d(1, 1, k, stride=s), id="conv3d"),
        pytest.param(lambda k, s: nn.MaxPool2d(k, stride=s), id="maxpool2d"),
        pytest.param(lambda k, s: nn.AvgPool2d(k, stride=s), id="avgpool2d"),
    ]
)
def base_module(request, kernel_size, stride):
    return request.param(kernel_size, stride)


def test_construct(base_module, padding_mode):
    DynamicSamePad(base_module, padding_mode)


def test_construct_overrides(base_module, padding_mode, kernel_size, stride):
    DynamicSamePad(base_module, padding_mode, kernel_size=kernel_size, stride=stride)


@pytest.mark.parametrize(
    "shape",
    [
        (32, 32),
        (9, 9),
        (11, 11),
    ],
)
@pytest.mark.parametrize("override", [True, False])
def test_forward(padding_mode, kernel_size, stride, shape, override):
    base_module = nn.Conv2d(1, 1, kernel_size, stride=stride)
    if not override:
        layer = DynamicSamePad(base_module, padding_mode)
    else:
        layer = DynamicSamePad(base_module, padding_mode, kernel_size=kernel_size, stride=stride)
    inputs = torch.rand(1, 1, *shape, requires_grad=True)

    output = layer(inputs)
    assert isinstance(output, Tensor)
    assert tuple(output.shape[:2]) == (1, 1)
    assert tuple(output.shape[2:]) == tuple([ceil(x / stride) for x in shape])
    assert output.requires_grad


def test_repr():
    layer = DynamicSamePad(nn.Conv2d(1, 1, 3))
    print(layer)


class TestScript(TorchScriptTestMixin):
    @pytest.fixture
    def model(self):
        return DynamicSamePad(nn.Conv2d(1, 1, 3, stride=2))

    @pytest.fixture
    def data(self):
        return torch.rand(1, 1, 11, 11)


class TestDynamicSamePadPatch:
    @pytest.fixture
    def module(self):
        _ = nn.Sequential(
            nn.Conv2d(1, 1, 3),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=1),
            nn.Conv2d(1, 1, 5, stride=2),
            nn.ConvTranspose2d(1, 1, 2, stride=2),
        )
        return _

    @pytest.mark.parametrize(
        "shape",
        [
            (11, 11),
            (8, 8),
            (12, 12),
        ],
    )
    def test_patch(self, module, shape):
        torch.random.manual_seed(42)
        inputs = torch.rand(1, 1, *shape)
        patch_dynamic_same_pad(module)
        output = module(inputs)

        expected_shape = tuple([ceil(x / 2) * 2 for x in shape])
        assert tuple(output.shape[-2:]) == expected_shape


class TestMatchShapes(TorchScriptTestMixin):
    @pytest.fixture
    def model(self):
        return MatchShapes()

    @pytest.mark.parametrize("strategy", ["crop", "pad"])
    @pytest.mark.parametrize(
        "shape1,shape2",
        [
            pytest.param((10, 10), (11, 11), id="case1"),
            pytest.param((9, 9), (13, 13), id="case2"),
            pytest.param((9,), (13, 13), id="case3", marks=pytest.mark.xfail(raises=RuntimeError)),
            pytest.param((5, 5), (10, 10), id="case4"),
        ],
    )
    def test_match_shapes(self, shape1, shape2, strategy):
        torch.random.manual_seed(42)
        t1 = torch.rand(1, 1, *shape1)
        t2 = torch.rand(1, 1, *shape2)
        layer = MatchShapes(strategy=strategy)
        t1, t2 = layer([t1, t2])

        assert t1.shape == t2.shape
        if strategy == "crop":
            assert tuple(t2.shape[2:]) == shape1
        elif strategy == "pad":
            assert tuple(t1.shape[2:]) == shape2

    @pytest.mark.parametrize("strategy", ["crop", "pad"])
    @pytest.mark.parametrize(
        "shape1,shape2",
        [
            pytest.param((5, 5), (10, 10), id="case1"),
            pytest.param((10, 10), (5, 5), id="case2"),
        ],
    )
    def test_warns_on_major_resize(self, shape1, shape2, strategy):
        torch.random.manual_seed(42)
        t1 = torch.rand(1, 1, *shape1)
        t2 = torch.rand(1, 1, *shape2)
        layer = MatchShapes(strategy=strategy, warn_pct_change=0.5)
        with pytest.warns(UserWarning):
            t1, t2 = layer([t1, t2])

    @pytest.mark.parametrize(
        "strategy,shape1,shape2,target",
        [
            pytest.param("crop", (10, 10), (12, 12), (9, 9), id="case1"),
            pytest.param("pad", (10, 10), (12, 12), (12, 12), id="case2"),
            pytest.param("pad", (10, 10), (12, 12), (13, 13), id="case3"),
            pytest.param("crop", (10, 10), (12, 12), (10, 10), id="case4"),
            pytest.param("crop", (10, 10), (12, 12), (10,), id="case5", marks=pytest.mark.xfail(raises=RuntimeError)),
            pytest.param("crop", (10, 10), (12, 12), (13, 13), id="case6"),
            pytest.param("pad", (10, 10), (12, 12), (9, 9), id="case7"),
            pytest.param("pad", (10, 10), (12, 12), (11, 11), id="case8"),
            pytest.param("crop", (58, 58), None, (57, 57), id="case9"),
            pytest.param("pad", (58, 58), None, (57, 57), id="case10"),
        ],
    )
    def test_explicit_shape(self, shape1, shape2, strategy, target):
        torch.random.manual_seed(42)
        layer = MatchShapes(strategy=strategy)
        t1 = torch.rand(1, 1, *shape1)
        if shape2 is not None:
            t2 = torch.rand(1, 1, *shape2)
            t1, t2 = layer([t1, t2], target)
        else:
            t1, t2 = layer([t1], target)[0], None

        if shape2 is not None:
            assert t1.shape == t2.shape
            assert tuple(t2.shape[2:]) == target
        assert tuple(t1.shape[2:]) == target

    def test_repr(self):
        layer = MatchShapes("pad")
        print(layer)

    def test_ignore_channels(self):
        torch.random.manual_seed(42)
        t1 = torch.rand(1, 3, 20, 20)
        t2 = torch.rand(1, 1, 17, 17)
        layer = MatchShapes(strategy="crop", ignore_channels=True)
        t1, t2 = layer([t1, t2])
        assert t1.shape[0] == t2.shape[0]
        assert t1.shape[1] == 3
        assert t2.shape[1] == 1
        assert t1.shape[2:] == t2.shape[2:]

    @pytest.mark.parametrize("strategy", ["crop", "pad"])
    def test_match_multiple_pad(self, strategy):
        torch.random.manual_seed(42)
        t1 = torch.rand(1, 3, 19, 20)
        t2 = torch.rand(1, 1, 17, 22)
        t3 = torch.rand(1, 1, 13, 14)
        layer = MatchShapes(strategy=strategy, ignore_channels=True)

        if strategy == "crop":
            shape = (13, 14)
        else:
            shape = (19, 22)

        t1, t2, t3 = layer([t1, t2, t3])
        assert tuple(t1.shape) == (1, 3, *shape)
        assert tuple(t2.shape) == (1, 1, *shape)
        assert tuple(t2.shape) == (1, 1, *shape)

    @pytest.mark.parametrize(
        "padding_mode,fill_value",
        [
            pytest.param("constant", 1, id="constant"),
            pytest.param("reflect", 0, id="reflect"),
            pytest.param("replicate", 0, id="replicate"),
            pytest.param("mean", 0, id="mean"),
            pytest.param("var_mean", 0, id="var_mean"),
        ],
    )
    def test_padding_mode(self, padding_mode, fill_value):
        torch.random.manual_seed(42)
        layer = MatchShapes(strategy="pad", padding_mode=padding_mode, fill_value=fill_value)
        t1_in = torch.rand(1, 1, 10, 10)
        t2_in = torch.rand(1, 1, 12, 12)

        t1, t2 = layer([t1_in, t2_in])

        assert t1.shape == t2.shape
        assert tuple(t2.shape[2:]) == (12, 12)
        assert tuple(t1.shape[2:]) == (12, 12)

        if padding_mode == "mean":
            assert torch.allclose(t1.mean(), t1_in.mean())
            assert torch.allclose(t2.mean(), t2_in.mean())
        if padding_mode == "var_mean":
            t1_var, t1_mean = torch.var_mean(t1_in)
            t1_var_new, t1_mean_new = torch.var_mean(t1)
            assert torch.allclose(t1_mean, t1_mean_new, atol=3e-2)
            assert torch.allclose(t1_var, t1_var_new, atol=2e-2)

            t2_var, t2_mean = torch.var_mean(t2_in)
            t2_var_new, t2_mean_new = torch.var_mean(t2)
            assert torch.allclose(t2_mean, t2_mean_new, atol=3e-2)
            assert torch.allclose(t2_var, t2_var_new, atol=2e-2)
