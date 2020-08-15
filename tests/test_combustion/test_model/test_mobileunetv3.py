#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gc

import pytest
import torch
from torch import Tensor

from combustion.models import MobileUnet1d, MobileUnet2d, MobileUnet3d
from combustion.nn import MobileNetBlockConfig
from combustion.testing import TorchScriptTestMixin


@pytest.mark.ci_skip
class MobileUnetBaseTest(TorchScriptTestMixin):
    @pytest.fixture
    def model_type(self):
        raise NotImplementedError()

    @pytest.fixture
    def data(self):
        raise NotImplementedError()

    @pytest.fixture
    def model(self, model_type):
        block = MobileNetBlockConfig(1, 1, kernel_size=3)
        model = model_type.from_identical_blocks(block, in_channels=1, levels=[1, 2, 3])
        yield model
        del model
        gc.collect()

    def test_forward(self, model, data):
        output = model(data)
        assert isinstance(output, Tensor)
        assert output.shape[2:] == data.shape[2:]

    def test_backward(self, model, data):
        output = model(data)
        scalar = output.sum()
        scalar.backward()


class TestMobileUnet1d(MobileUnetBaseTest):
    @pytest.fixture
    def model_type(self):
        return MobileUnet1d

    @pytest.fixture
    def data(self):
        torch.random.manual_seed(42)
        return torch.rand(1, 1, 227, requires_grad=True)


class TestMobileUnet2d(MobileUnetBaseTest):
    @pytest.fixture
    def model_type(self):
        return MobileUnet2d

    @pytest.fixture
    def data(self):
        torch.random.manual_seed(42)
        return torch.rand(1, 1, 227, 227, requires_grad=True)


class TestMobileUnet3d(MobileUnetBaseTest):
    @pytest.fixture
    def model_type(self):
        return MobileUnet3d

    @pytest.fixture
    def data(self):
        torch.random.manual_seed(43)
        return torch.rand(1, 1, 100, 100, 100, requires_grad=True)
