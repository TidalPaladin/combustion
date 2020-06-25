#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import torch
import torch.nn as nn

from combustion.testing import TorchScriptTestMixin, TorchScriptTraceTestMixin


class Model(nn.Module):
    def __init__(self, in_features, out_features, kernel):
        super(Model, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel = kernel
        self.conv = nn.Conv2d(in_features, out_features, kernel, padding=(kernel // 2))

    def forward(self, x):
        return self.conv(x)


class TestTorchScriptTestMixin(TorchScriptTestMixin):
    @pytest.fixture
    def model(self):
        return Model(1, 10, 3)


class TestTorchScriptTraceTestMixin(TorchScriptTraceTestMixin):
    @pytest.fixture
    def model(self):
        return Model(1, 10, 3)

    @pytest.fixture
    def data(self):
        return torch.rand(2, 1, 10, 10)
