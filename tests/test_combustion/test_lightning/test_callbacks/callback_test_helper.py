#!/usr/bin/env python
# -*- coding: utf-8 -*-

import inspect

import pytest
import pytorch_lightning as pl
import torch

from combustion.testing import assert_equal
from combustion.vision import to_8bit


def check_call(call, name, img, step, as_uint8=True):
    if as_uint8:
        img = to_8bit(img, same_on_batch=True)
    assert call.args[0] == name
    assert torch.allclose(call.args[1], img, atol=1)
    assert call.args[2] == step


def assert_calls_equal(call1, call2, **kwargs):
    __tracebackhide__ = True
    call1 = tuple(call1.args) if type(call1) != tuple else call1
    call2 = tuple(call2.args) if type(call2) != tuple else call2
    for arg1, arg2 in zip(call1, call2):
        assert_equal(arg1, arg2, **kwargs)


class BaseAttributeCallbackTest:
    callback_cls: type

    @pytest.fixture
    def attr(self):
        raise NotImplementedError("attr fixture")

    @pytest.fixture
    def trainer(self, tmp_path, capsys):
        trainer = pl.Trainer(default_root_dir=tmp_path)
        return trainer

    # training, validation, or testing mode
    @pytest.fixture(params=["train", "val", "test"])
    def mode(self, request):
        return request.param

    # training, validation, or testing mode
    @pytest.fixture(params=["epoch", "step"])
    def hook(self, request):
        return request.param

    @pytest.fixture
    def callback(self, mocker, request, hook):
        cls = self.callback_cls
        init_signature = inspect.signature(cls)
        defaults = {
            k: v.default for k, v in init_signature.parameters.items() if v.default is not inspect.Parameter.empty
        }

        if hasattr(request, "param"):
            defaults.update(request.param)
        defaults["hook"] = hook

        callback = cls(**defaults)

        # spy the callback function so we can inspect calls
        callback.callback_fn = mocker.spy(callback, "callback_fn")
        return callback

    @pytest.fixture
    def model(self, request, mocker, callback, attr, mode, hook, trainer):
        if hasattr(request, "param"):
            step = request.param.pop("step", 10)
            epoch = request.param.pop("epoch", 1)
            batch_idx = request.param.pop("epoch", 0)
        else:
            step = 10
            epoch = 1
            batch_idx = 0

        model = mocker.MagicMock(name="module")
        model.current_epoch = epoch
        model.global_step = step
        if callback.attr_name is not None:
            setattr(model, callback.attr_name, attr)

        _hook = "batch" if hook == "step" else "epoch"
        _mode = "validation" if "val" in mode else mode
        func_name = f"on_{_mode}_{_hook}_end"
        if _hook == "batch":
            model.batch_idx = batch_idx
            callback.trigger = lambda: getattr(callback, func_name)(trainer, model, None, None, batch_idx)
        else:
            model.batch_idx = None
            callback.trigger = lambda: getattr(callback, func_name)(trainer, model, None)
        callback.on_pretrain_routine_start(trainer, model)

        return model

    def test_repr(self, callback):
        print(callback)

    @pytest.mark.usefixtures("model", "mode", "hook")
    def test_callback_fn_call(self, callback):
        callback.trigger()
        func = callback.callback_fn
        func.assert_called_once()
        assert callback.counter == 1

    @pytest.mark.usefixtures("model")
    @pytest.mark.parametrize(
        "callback",
        [
            pytest.param(dict(max_calls=5), id="max_calls=5"),
            pytest.param(dict(max_calls=10), id="max_calls=10"),
            pytest.param(dict(max_calls=20), id="max_calls=20"),
        ],
        indirect=True,
    )
    def test_max_calls(self, callback):
        num_steps = 20
        for _ in range(num_steps):
            callback.trigger()

        limit = callback.max_calls
        expected = min(limit, num_steps) if limit is not None else num_steps
        assert callback.callback_fn.call_count == expected

    @pytest.mark.parametrize(
        "model",
        [
            pytest.param(dict(epoch=1, step=1), id="epoch=1,step=1"),
            pytest.param(dict(epoch=1, step=10), id="epoch=1,step=10"),
            pytest.param(dict(epoch=10, step=1), id="epoch=10,step=1"),
            pytest.param(dict(epoch=20, step=20), id="epoch=20,step=20"),
            pytest.param(dict(epoch=32, step=32), id="epoch=32,step=32"),
            pytest.param(dict(epoch=1, step=100, batch_idx=2), id="epoch=32,step=32,batch_idx=2"),
        ],
        indirect=True,
    )
    @pytest.mark.parametrize(
        "callback",
        [
            pytest.param(dict(interval=10, epoch_counter=False), id="interval=10,epoch_counter=False"),
            pytest.param(dict(interval=10, epoch_counter=True), id="interval=10,epoch_counter=True"),
            pytest.param(dict(interval=1, epoch_counter=False), id="interval=1,epoch_counter=False"),
            pytest.param(dict(interval=1, epoch_counter=True), id="interval=1,epoch_counter=True"),
        ],
        indirect=True,
    )
    def test_log_interval(self, callback, model):
        callback.trigger()
        epoch = model.current_epoch
        step = model.global_step
        count_from_epoch = callback.epoch_counter
        interval = callback.interval

        if count_from_epoch and epoch % interval == 0:
            should_log = True
        elif not count_from_epoch and step % interval == 0:
            should_log = True
        else:
            should_log = False

        if should_log:
            callback.callback_fn.assert_called()
        else:
            callback.callback_fn.assert_not_called()
