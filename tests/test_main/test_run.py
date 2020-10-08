#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import runpy
import sys

import pytest
import pytorch_lightning as pl

from combustion import MultiRunError


pytest.importorskip("torchvision", reason="test requires torchvision")


@pytest.fixture(autouse=False)
def set_caplog(caplog):
    caplog.set_level(logging.CRITICAL, logger="__main__")
    caplog.set_level(logging.CRITICAL, logger="combustion")
    caplog.set_level(logging.CRITICAL)


@pytest.mark.parametrize("deterministic", [True, False])
def test_fast_dev_run(mocker, deterministic):
    m = mocker.MagicMock(spec_set=pl.Trainer())
    mocker.patch("pytorch_lightning.Trainer", return_value=m)
    sys.argv = [
        sys.argv[0],
        "trainer=test",
        "trainer.catch_exceptions=False",
        f"trainer.params.deterministic={deterministic}",
    ]
    runpy.run_module("examples.basic", run_name="__main__", alter_sys=True)
    m.fit.assert_called_once()
    m.test.assert_called_once()


def test_skip_train(mocker):
    m = mocker.MagicMock(spec_set=pl.Trainer())
    mocker.patch("pytorch_lightning.Trainer", return_value=m)
    sys.argv = [
        sys.argv[0],
        "trainer=test",
        "trainer.catch_exceptions=False",
        "trainer.test_only=True",
    ]
    runpy.run_module("examples.basic", run_name="__main__", alter_sys=True)
    m.fit.assert_not_called()
    m.test.assert_called_once()


def test_skip_loading_train_dataset(mocker):
    torchvision = pytest.importorskip("torchvision")
    m = mocker.spy(torchvision.datasets, "FakeData")
    sys.argv = [
        sys.argv[0],
        "trainer=test",
        "trainer.catch_exceptions=False",
        "trainer.test_only=True",
    ]
    runpy.run_module("examples.basic", run_name="__main__", alter_sys=True)
    m.assert_called_once()


@pytest.mark.parametrize("resume_from_checkpoint", ["foo/bar.ckpt", None])
def test_load_checkpoint(mocker, resume_from_checkpoint):
    checkpoint = "path/to/checkpoint.ckpt"
    m = mocker.MagicMock(spec_set=pl.LightningModule.load_from_checkpoint)
    mocker.patch("pytorch_lightning.LightningModule.load_from_checkpoint", m)
    mocker.patch("pytorch_lightning.Trainer.fit")
    mocker.patch("pytorch_lightning.Trainer.test")
    sys.argv = [
        sys.argv[0],
        "trainer=test",
        "trainer.catch_exceptions=False",
        f"trainer.load_from_checkpoint={checkpoint}",
    ]
    if resume_from_checkpoint is not None:
        sys.argv += [f"trainer.params.resume_from_checkpoint={resume_from_checkpoint}"]

    runpy.run_module("examples.basic", run_name="__main__", alter_sys=True)

    if resume_from_checkpoint is not None:
        m.assert_not_called()
    else:
        m.assert_called_once_with(checkpoint)
    pl.Trainer.fit.assert_called_once()
    pl.Trainer.test.assert_called_once()


def test_initialize_checks_hydra_version(mocker):
    mocker.patch("hydra.__version__", new="1.0.0rc1")
    sys.argv = [sys.argv[0], "-m", "trainer=test", "model.params.batch_size=8,32"]
    with pytest.raises(ImportError):
        runpy.run_module("examples.basic", run_name="__main__", alter_sys=True)


# NOTE: for some reason, this test needs to run before the multirun tests


def test_lr_auto_find():
    sys.argv = [
        sys.argv[0],
        "trainer=test",
        "trainer.params.auto_lr_find=True",
        "trainer.params.fast_dev_run=False",
        "model.params.batch_size=8",
    ]
    runpy.run_module("examples.basic", run_name="__main__", alter_sys=True)


def test_multirun():
    sys.argv = [sys.argv[0], "-m", "trainer=test", "model.params.batch_size=8,32"]
    runpy.run_module("examples.basic", run_name="__main__", alter_sys=True)


def test_multirun_from_yaml():
    sys.argv = [sys.argv[0], "-m", "trainer=test", "sweeper=sweep1"]
    runpy.run_module("examples.basic", run_name="__main__", alter_sys=True)


def test_multirun_handles_exception():
    sys.argv = [sys.argv[0], "-m", "trainer=test", "model.params.batch_size=-1, 8"]
    with pytest.raises(MultiRunError):
        runpy.run_module("examples.basic", run_name="__main__", alter_sys=True)


@pytest.mark.parametrize("ex", [KeyboardInterrupt, SystemExit])
def test_multirun_abort(mocker, ex):
    mocker.patch("pytorch_lightning.Trainer.fit", side_effect=ex)
    sys.argv = [sys.argv[0], "-m", "trainer=test", "model.params.batch_size=-1, 8"]
    with pytest.raises(ex):
        runpy.run_module("examples.basic", run_name="__main__", alter_sys=True)


@pytest.mark.skip
def test_dev_run():
    sys.argv = [sys.argv[0], "trainer=test", "trainer.params.fast_dev_run=False"]
    runpy.run_module("examples.basic", run_name="__main__", alter_sys=True)
