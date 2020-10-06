#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import torch

from combustion.nn import SqueezeExcite1d, SqueezeExcite2d, SqueezeExcite3d
from combustion.testing import TorchScriptTestMixin, TorchScriptTraceTestMixin


class TestSqueezeExcite1d(TorchScriptTestMixin, TorchScriptTraceTestMixin):
    @pytest.fixture(params=[True, False])
    def global_pool(self, request):
        return request.param

    @pytest.fixture(params=["avg", "max"])
    def pool_type(self, request):
        return request.param

    @pytest.fixture
    def model_type(self):
        return SqueezeExcite1d

    @pytest.fixture
    def model(self, model_type, global_pool, pool_type):
        return model_type(4, 2, global_pool=global_pool, pool_type=pool_type)

    @pytest.fixture
    def data(self):
        torch.random.manual_seed(42)
        return torch.rand(1, 4, 32)

    def test_forward(self, model, data):
        output = model(data)
        assert output.ndim == data.ndim
        assert output.shape[0:2] == data.shape[0:2]

    def test_backward(self, model, data):
        output = model(data)
        scalar = output.sum()
        scalar.backward()

    def test_limits_squeeze_ratio(self, model_type, data):
        model = model_type(4, 100)
        output = model(data)
        assert output.ndim == data.ndim
        assert output.shape[0:2] == data.shape[0:2]


class TestSqueezeExcite2d(TestSqueezeExcite1d):
    @pytest.fixture(params=["avg", "max", torch.nn.AdaptiveAvgPool2d])
    def pool_type(self, request):
        return request.param

    @pytest.fixture
    def model_type(self):
        return SqueezeExcite2d

    @pytest.fixture
    def data(self):
        torch.random.manual_seed(42)
        return torch.rand(1, 4, 32, 32)


class TestSqueezeExcite3d(TestSqueezeExcite1d):
    @pytest.fixture
    def model_type(self):
        return SqueezeExcite3d

    @pytest.fixture
    def data(self):
        torch.random.manual_seed(42)
        return torch.rand(1, 4, 32, 32, 9)
