#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import pytest
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch import Tensor
from torch.jit import ScriptModule

from combustion.lightning import TorchScriptCallback


class BadModel(pl.LightningModule):
    def __init__(self, in_features, out_features, kernel):
        super(BadModel, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel = kernel
        self.conv = nn.Conv2d(in_features, out_features, kernel, padding=(kernel // 2))

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class Model(BadModel):
    _device: torch.device


class TestTorchScriptCallback:
    @pytest.fixture
    def model(self):
        return Model(10, 10, 3)

    @pytest.fixture(scope="class")
    def trainer(self):
        return pl.Trainer(fast_dev_run=True)

    @pytest.fixture
    def data(self):
        return torch.rand(1, 10, 32, 32)

    @pytest.fixture
    def path(self, tmp_path):
        return os.path.join(tmp_path, "model.pth")

    def test_script_exported(self, model, trainer, path):
        callback = TorchScriptCallback(path)
        callback.on_train_end(trainer, model)
        assert os.path.isfile(path)

    def test_exported_script_is_loadable(self, model, trainer, path):
        callback = TorchScriptCallback(path)
        callback.on_train_end(trainer, model)
        loaded = torch.jit.load(path)
        assert isinstance(loaded, ScriptModule)

    def test_trace_exported(self, model, trainer, path, data):
        callback = TorchScriptCallback(path, True, data)
        callback.on_train_end(trainer, model)
        assert os.path.isfile(path)

    def test_exported_trace_is_loadable(self, model, trainer, path, data):
        callback = TorchScriptCallback(path, True, data)
        callback.on_train_end(trainer, model)
        loaded = torch.jit.load(path)
        assert isinstance(loaded, ScriptModule)

    def test_exception_on_device_type_ellipsis(self, trainer, path):
        model = BadModel(10, 10, 3)
        callback = TorchScriptCallback(path)
        with pytest.raises(RuntimeError):
            callback.on_train_end(trainer, model)

    def test_trace_example_input_array(self, model, trainer, path, data):
        model.example_input_array = data
        callback = TorchScriptCallback(path, True)
        callback.on_train_end(trainer, model)
        assert os.path.isfile(path)
