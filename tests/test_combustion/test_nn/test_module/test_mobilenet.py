#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import torch

from combustion.nn import MobileNetConvBlock1d, MobileNetConvBlock2d, MobileNetConvBlock3d
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
