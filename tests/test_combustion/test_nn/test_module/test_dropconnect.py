#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import torch

from combustion.nn import DropConnect
from combustion.testing import TorchScriptTestMixin, TorchScriptTraceTestMixin


@pytest.mark.filterwarnings("ignore::torch.jit.TracerWarning")
class TestDropConnect(TorchScriptTestMixin, TorchScriptTraceTestMixin):
    @pytest.fixture(params=["train", "eval"])
    def mode(self, request):
        return request.param

    @pytest.fixture(params=[1, 4])
    def batch_size(self, request):
        return request.param

    @pytest.fixture
    def model(self, mode):
        layer = DropConnect(0.3)
        if mode == "train":
            layer.train()
        else:
            layer.eval()
        return layer

    @pytest.fixture(
        params=[
            pytest.param(1, id="1d"),
            pytest.param(2, id="2d"),
            pytest.param(3, id="3d"),
        ]
    )
    def data(self, request, batch_size):
        torch.random.manual_seed(42)
        return torch.rand(batch_size, 4, *([32] * request.param), requires_grad=True)

    def test_forward(self, mode, model, data):
        output = model(data)
        if mode == "train":
            assert model.training
            assert output.shape == data.shape
            assert output is not data
        else:
            assert not model.training
            assert output is data

    def test_backward(self, model, data):
        output = model(data)
        scalar = output.sum()
        scalar.backward()
