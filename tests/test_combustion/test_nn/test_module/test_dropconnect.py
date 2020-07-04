#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import torch

from combustion.nn import DropConnect
from combustion.testing import TorchScriptTestMixin, TorchScriptTraceTestMixin


class TestDropConnect(TorchScriptTestMixin, TorchScriptTraceTestMixin):
    @pytest.fixture
    def model(self):
        return DropConnect(0.3)

    @pytest.fixture(
        params=[pytest.param(1, id="1d"), pytest.param(2, id="2d"), pytest.param(3, id="3d"),]
    )
    def data(self, request):
        torch.random.manual_seed(42)
        return torch.rand(1, 4, *([32] * request.param), requires_grad=True)

    def test_forward(self, model, data):
        output = model(data)
        assert output.shape == data.shape

    def test_backward(self, model, data):
        output = model(data)
        scalar = output.sum()
        scalar.backward()
