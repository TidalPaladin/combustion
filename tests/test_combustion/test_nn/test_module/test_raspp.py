#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gc

import pytest
import torch
from torch import Tensor

from combustion.nn import RASPPLite2d
from combustion.testing import TorchScriptTestMixin


class RASPPBaseTest(TorchScriptTestMixin):
    @pytest.fixture
    def model_class(self):
        raise NotImplementedError()

    @pytest.fixture
    def data(self):
        raise NotImplementedError()

    @pytest.fixture
    def input_channels(self):
        return 32

    @pytest.fixture
    def residual_channels(self):
        return 4

    @pytest.fixture
    def output_channels(self):
        return 8

    @pytest.fixture
    def num_classes(self):
        return 2

    @pytest.fixture
    def model(self, model_class, input_channels, residual_channels, output_channels, num_classes):
        model = model_class(
            input_channels, residual_channels, output_channels, num_classes, pool_kernel=2, pool_stride=1
        )
        yield model
        del model
        gc.collect()

    def test_init(self, model_class, input_channels, residual_channels, output_channels, num_classes):
        model = model_class(input_channels, residual_channels, output_channels, num_classes)
        del model
        gc.collect()

    def test_forward(self, model, num_classes, data):
        inputs = data
        # if torch.cuda.is_available():
        #    layer = model.cuda()
        #    output = layer([x.cuda() for x in inputs])
        # else:
        layer = model
        output = layer(inputs)

        assert isinstance(output, Tensor)
        assert output.shape[1] == num_classes
        assert output.requires_grad

    def test_backward(self, model, data):
        inputs = data
        for x in inputs:
            x.requires_grad = True

        # if torch.cuda.is_available():
        #    layer = model.cuda()
        #    output = layer([x.cuda() for x in inputs])
        # else:
        layer = model
        output = layer(inputs)

        scalar: Tensor = sum([x.sum() for x in output])  # type: ignore
        scalar.backward()


class TestRASPP2d(RASPPBaseTest):
    @pytest.fixture
    def model_class(self):
        return RASPPLite2d

    @pytest.fixture
    def data(self, input_channels, residual_channels):
        batch_size = 2
        result = []

        t = torch.rand(batch_size, residual_channels, 32, 32, requires_grad=True)
        result.append(t)
        t = torch.rand(batch_size, input_channels, 4, 4, requires_grad=True)
        result.append(t)

        return result
