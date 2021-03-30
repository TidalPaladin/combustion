#!/usr/bin/env python
# -*- coding: utf-8 -*-


from pathlib import Path

import pytest
import torch

from combustion.lightning.callbacks import SaveTensors

from .callback_test_helper import BaseAttributeCallbackTest, assert_calls_equal


class TestSaveTensors(BaseAttributeCallbackTest):
    callback_cls = SaveTensors

    @pytest.fixture
    def attr(self):
        torch.random.manual_seed(42)
        return torch.rand(2, 3, 32, 32)

    @pytest.fixture
    def expected_calls(self, attr, model, mode, callback):
        if not hasattr(model, callback.attr_name):
            return []

        img = [data]
        name = [
            f"{mode}/{callback.name}",
        ]

        step = [model.current_epoch if callback.epoch_counter else model.global_step] * len(name)
        expected = [(n, i, s) for n, i, s in zip(name, img, step)]
        return expected

    def test_saved_tensor(self, mocker, model, mode, hook, callback, tmp_path, attr, trainer):
        spy = mocker.spy(torch, "save")
        trainer.default_root_dir = tmp_path
        callback.trigger()
        path = Path(tmp_path, mode, f"{callback.attr_name}_{callback.read_step(model)}.pth")
        spy.assert_called_once()
        assert_calls_equal(spy.call_args, (attr, path), atol=1e-4)
