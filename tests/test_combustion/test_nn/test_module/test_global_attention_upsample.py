#!/usr/bin/env python
# -*- coding: utf-8 -*-
import gc

import pytest
import torch
from torch import Tensor

from combustion.nn import AttentionUpsample1d, AttentionUpsample2d, AttentionUpsample3d
from combustion.testing import TorchScriptTestMixin


class AttentionUpsampleBaseTest(TorchScriptTestMixin):
    @pytest.fixture
    def model_class(self):
        raise NotImplementedError()

    @pytest.fixture
    def data(self, request, levels, num_classes):
        raise NotImplementedError()

    @pytest.fixture
    def low_filters(self):
        return 8

    @pytest.fixture
    def high_filters(self):
        return 10

    @pytest.fixture
    def output_filters(self):
        return 2

    @pytest.fixture(params=[True, False])
    def pool(self, request):
        return request.param

    @pytest.fixture
    def model(self, model_class, low_filters, high_filters, output_filters, pool):
        model = model_class(low_filters, high_filters, output_filters, pool=pool)
        yield model
        del model
        gc.collect()

    def test_init(self, model_class, low_filters, high_filters, output_filters, pool):
        model = model_class(low_filters, high_filters, output_filters, pool=pool)
        del model
        gc.collect()

    def test_forward(self, model, data, output_filters):
        low, high = data
        layer = model
        output = layer(low, high)

        assert isinstance(output, Tensor)
        assert output.shape[1] == output_filters
        assert output.shape[2:] == low.shape[2:]
        assert output.requires_grad

    def test_backward(self, model, data):
        low, high = data
        low.requires_grad = True
        high.requires_grad = True

        layer = model
        output = layer(low, high)

        scalar = sum([x.sum() for x in output])
        scalar.backward()


class TestAttentionUpsample2d(AttentionUpsampleBaseTest):
    @pytest.fixture
    def model_class(self, request):
        return AttentionUpsample2d

    @pytest.fixture(params=["same_shape", "different_shape"])
    def data(self, request, low_filters, high_filters):
        batch_size = 2

        low_shape = (batch_size, low_filters, 32, 32)
        if request.param == "different_shape":
            high_shape = (batch_size, high_filters, 16, 16)
        else:
            high_shape = (batch_size, high_filters, 32, 32)

        low = torch.rand(*low_shape, requires_grad=True)
        high = torch.rand(*high_shape, requires_grad=True)
        return low, high


class TestAttentionUpsample3d(AttentionUpsampleBaseTest):
    @pytest.fixture
    def model_class(self, request):
        return AttentionUpsample3d

    @pytest.fixture(params=["same_shape", "different_shape"])
    def data(self, request, low_filters, high_filters):
        batch_size = 2

        low_shape = (batch_size, low_filters, 32, 32, 32)
        if request.param == "different_shape":
            high_shape = (batch_size, high_filters, 16, 16, 16)
        else:
            high_shape = (batch_size, high_filters, 32, 32, 32)

        low = torch.rand(*low_shape, requires_grad=True)
        high = torch.rand(*high_shape, requires_grad=True)
        return low, high


class TestAttentionUpsample1d(AttentionUpsampleBaseTest):
    @pytest.fixture
    def model_class(self, request):
        return AttentionUpsample1d

    @pytest.fixture(params=["same_shape", "different_shape"])
    def data(self, request, low_filters, high_filters):
        batch_size = 2

        low_shape = (batch_size, low_filters, 32)
        if request.param == "different_shape":
            high_shape = (batch_size, high_filters, 16)
        else:
            high_shape = (batch_size, high_filters, 32)

        low = torch.rand(*low_shape, requires_grad=True)
        high = torch.rand(*high_shape, requires_grad=True)
        return low, high
