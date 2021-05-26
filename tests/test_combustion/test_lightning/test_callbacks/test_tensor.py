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

        img = [attr]
        name = [
            f"{mode}/{callback.name}",
        ]

        step = [model.current_epoch if callback.epoch_counter else model.global_step] * len(name)
        expected = [(n, i, s) for n, i, s in zip(name, img, step)]
        return expected

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
            pytest.param(dict(output_format="pth"), id="pth"),
            pytest.param(dict(output_format="mat"), id="mat"),
            pytest.param(dict(output_format="csv"), id="csv"),
            pytest.param(dict(output_format="foo"), marks=pytest.mark.xfail(raises=ValueError), id="bad_format"),
        ],
        indirect=True,
    )
    def test_save_tensor(self, mocker, model, mode, hook, callback, tmp_path, attr, trainer):
        path = Path(
            tmp_path,
            "lightning_logs",
            "version_0",
            "saved_tensors",
            mode,
            f"{callback.attr_name}",
            f"{callback.read_step_as_str(model, model.batch_idx)}",
        ).with_suffix(".pth")
        path = path.with_suffix(f".{callback.output_format}")
        callback.ignore_errors = False

        if callback.output_format == "pth":
            spy = mocker.spy(torch, "save")
            call = (attr, path)

        elif callback.output_format == "mat":
            sio = pytest.importorskip("scipy.io", reason="test requires scipy")
            spy = mocker.spy(sio, "savemat")
            call = (path, {"tensor": attr.cpu().numpy()})

        elif callback.output_format == "csv":
            spy = None
            call = None
            setattr(model, callback.attr_name, attr[0, 0])

        else:
            raise RuntimeError(f"output_format {callback.output_format}")

        # for csv, just test that the call succeeds
        if spy is not None:
            callback.trigger()
            spy.assert_called_once()
            assert_calls_equal(spy.call_args, call, atol=1e-4)

    @pytest.mark.parametrize(
        "callback",
        [
            pytest.param(dict(output_format=["pth", "mat"]), id="pth,mat"),
        ],
        indirect=True,
    )
    def test_save_multiple_tensors(self, mocker, model, mode, callback, tmp_path, attr):
        callback.ignore_errors = False
        sio = pytest.importorskip("scipy.io", reason="test requires scipy")
        path_torch = Path(
            tmp_path,
            "lightning_logs",
            "version_0",
            "saved_tensors",
            mode,
            f"{callback.attr_name}",
            f"{callback.read_step_as_str(model, model.batch_idx)}",
        ).with_suffix(".pth")
        path_mat = path_torch.with_suffix(".mat")
        spy_mat = mocker.spy(sio, "savemat")
        spy_torch = mocker.spy(torch, "save")
        call_torch = (attr, path_torch)
        call_mat = (path_mat, {"tensor": attr.cpu().numpy()})
        callback.trigger()
        spy_mat.assert_called_once()
        spy_torch.assert_called_once()
        assert_calls_equal(spy_torch.call_args, call_torch, atol=1e-4)
        assert_calls_equal(spy_mat.call_args, call_mat, atol=1e-4)
