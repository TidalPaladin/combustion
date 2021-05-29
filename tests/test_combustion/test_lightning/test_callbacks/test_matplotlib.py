#!/usr/bin/env python
# -*- coding: utf-8 -*-
import inspect
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from combustion.lightning.callbacks import MatplotlibCallback, PyplotSave

from .callback_test_helper import BaseAttributeCallbackTest, assert_calls_equal


class TestMatplotlibCallback(BaseAttributeCallbackTest):
    callback_cls = MatplotlibCallback

    @pytest.fixture
    def attr(self):
        fig = plt.figure()
        yield fig
        plt.close()

    @pytest.fixture
    def logger_func(self, model):
        return model.logger.experiment.add_figure

    @pytest.fixture
    def callback(self, request, mocker, hook):
        cls = self.callback_cls
        init_signature = inspect.signature(cls)
        defaults = {
            k: v.default for k, v in init_signature.parameters.items() if v.default is not inspect.Parameter.empty
        }

        if hasattr(request, "param"):
            name = request.param.get("name", "image")
            defaults.update(request.param)
        else:
            name = "image"

        defaults["name"] = name
        defaults["hook"] = hook
        callback = cls(**defaults)
        callback.callback_fn = mocker.spy(callback, "callback_fn")
        return callback

    @pytest.fixture
    def expected_calls(self, attr, model, mode, callback):
        if not hasattr(model, callback.attr_name):
            return []

        img = [attr]
        name = [
            f"{mode}/{callback.name}",
        ]

        step = [model.current_epoch if callback.epoch_counter else model.global_step] * len(name)
        expected = [(n, i, s) for n, i, s in zip(name, img, step)]
        return expected

    @pytest.mark.parametrize(
        "callback",
        [
            pytest.param(dict(name="foo"), id="name=foo"),
            pytest.param(dict(name="bar"), id="name=bar"),
        ],
        indirect=True,
    )
    def test_log_plot(self, callback, logger_func, expected_calls):
        callback.trigger()
        for actual, expected in zip(logger_func.mock_calls, expected_calls):
            assert_calls_equal(actual, expected)

    @pytest.mark.parametrize("relpath", [None, "subdir"])
    def test_save_plot(self, mocker, model, mode, callback, attr, relpath, tmp_path):
        subdir = Path(mode, callback.name, callback.read_step_as_str(model, model.batch_idx))
        if relpath is None:
            real_path = Path(tmp_path, "lightning_logs", "version_0", "saved_figures")
            init_path = None
        else:
            real_path = Path(tmp_path, relpath)
            init_path = real_path

        real_path = Path(real_path, subdir).with_suffix(".png")

        spy = mocker.spy(plt.Figure, "savefig")
        logger_func = PyplotSave(init_path)
        callback.log_fn = (logger_func,)
        callback.trigger()
        spy.assert_called_once_with(attr, str(real_path))
