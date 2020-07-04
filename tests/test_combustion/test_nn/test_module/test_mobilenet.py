#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import torch

from combustion.nn import MobileNetBlockConfig, MobileNetConvBlock1d, MobileNetConvBlock2d, MobileNetConvBlock3d
from combustion.testing import TorchScriptTestMixin, TorchScriptTraceTestMixin


class TestMobileNetConvBlock1d(TorchScriptTestMixin, TorchScriptTraceTestMixin):
    @pytest.fixture
    def model_type(self):
        return MobileNetConvBlock1d

    @pytest.fixture(
        params=[
            pytest.param((3, 1, 0.0, 1, 1, True), id="kernel=3,stride=1,dc=0.0,squeeze=1,excite=1,skip=False"),
            pytest.param((5, 2, 0.0, 2, 2, True), id="kernel=3,stride=1,dc=0.0,squeeze=1,excite=1,skip=True"),
            pytest.param((3, 1, 0.3, 8, 8, False), id="kernel=3,stride=1,dc=0.0,squeeze=1,excite=1,skip=False"),
        ]
    )
    def model(self, model_type, request):
        kernel, stride, dc, squeeze, excite, skip = request.param
        return model_type(
            4,
            4,
            kernel,
            stride=stride,
            drop_connect_rate=dc,
            squeeze_excite_ratio=squeeze,
            expand_ratio=excite,
            use_skipconn=skip,
        )

    @pytest.fixture
    def data(self):
        return torch.rand(1, 4, 32)

    def test_construct(self, model_type):
        model_type(4, 4, 3, drop_connect_rate=0.1, squeeze_excite_ratio=2, expand_ratio=2)

    def test_forward(self, model, data):
        output = model(data)
        assert isinstance(output, torch.Tensor)
        assert output.shape[:2] == data.shape[:2]
        if model._stride == 1:
            assert output.shape == data.shape
        else:
            assert list(output.shape[2:]) == [x // 2 for x in data.shape[2:]]

    def test_backward(self, model, data):
        output = model(data)
        scalar = output.sum()
        scalar.backward()


class TestMobileNetConvBlock2d(TestMobileNetConvBlock1d):
    @pytest.fixture
    def model_type(self):
        return MobileNetConvBlock2d

    @pytest.fixture
    def data(self):
        return torch.rand(1, 4, 32, 32)


class TestMobileNetConvBlock3d(TestMobileNetConvBlock1d):
    @pytest.fixture
    def model_type(self):
        return MobileNetConvBlock3d

    @pytest.fixture
    def data(self):
        return torch.rand(1, 4, 32, 32, 32)


class TestMobileNetBlockConfig:
    @pytest.fixture(params=[1, 2])
    def num_repeats(self, request):
        return request.param

    @pytest.fixture
    def config(self, num_repeats):
        return MobileNetBlockConfig(input_filters=4, output_filters=4, kernel_size=3, num_repeats=num_repeats)

    @pytest.fixture
    def data(self):
        torch.random.manual_seed(42)
        return torch.rand(1, 4, 32, 32)

    def test_get_1d_blocks(self, config, data, num_repeats):
        output = config.get_1d_blocks()
        if num_repeats > 1:
            assert isinstance(output, torch.nn.Sequential)
        else:
            assert isinstance(output, MobileNetConvBlock1d)

    def test_get_2d_blocks(self, config, data, num_repeats):
        output = config.get_2d_blocks()
        if num_repeats > 1:
            assert isinstance(output, torch.nn.Sequential)
        else:
            assert isinstance(output, MobileNetConvBlock2d)

    def test_get_3d_blocks(self, config, data, num_repeats):
        output = config.get_3d_blocks()
        if num_repeats > 1:
            assert isinstance(output, torch.nn.Sequential)
        else:
            assert isinstance(output, MobileNetConvBlock3d)
