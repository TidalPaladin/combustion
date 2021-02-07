#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

import pytest
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch import Tensor
from torch.jit import ScriptModule

from combustion.lightning.callbacks import CountMACs, TorchScriptCallback


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

    @pytest.mark.parametrize("training", [True, False])
    def test_module_set_to_eval_mode(self, model, trainer, path, mocker, training):
        if training:
            model.train()
        else:
            model.eval()
        spy = mocker.spy(model, "eval")
        callback = TorchScriptCallback(path)
        callback.on_train_end(trainer, model)

        if training:
            spy.assert_called()

    @pytest.mark.parametrize("training", [True, False])
    def test_module_training_state_unchanged(self, model, trainer, path, mocker, training):
        if training:
            model.train()
        else:
            model.eval()
        callback = TorchScriptCallback(path)
        callback.on_train_end(trainer, model)
        assert model.training == training


class TestCountMACs:
    @pytest.fixture(autouse=True, scope="class")
    def check_import(self):
        pytest.importorskip("thop", reason="test requires thop")

    @pytest.fixture
    def model(self):
        return Model(10, 10, 3)

    @pytest.fixture(scope="class")
    def trainer(self):
        return pl.Trainer(fast_dev_run=True)

    @pytest.fixture
    def data(self):
        return torch.rand(1, 10, 32, 32)

    def test_count_macs(self, model, trainer, data):
        callback = CountMACs(data)
        callback.on_train_start(trainer, model)

        assert hasattr(model, "macs_count")
        assert model.macs_count == 931840

    def test_count_params(self, model, trainer, data):
        callback = CountMACs(data)
        callback.on_train_start(trainer, model)

        assert hasattr(model, "param_count")
        assert model.param_count == 910

    def test_handles_no_input_data(self, model, trainer, data):
        callback = CountMACs()
        callback.on_train_start(trainer, model)
