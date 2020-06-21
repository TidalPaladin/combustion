#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import runpy
import sys

import pytest


@pytest.fixture(autouse=True)
def set_caplog(caplog):
    caplog.set_level(logging.CRITICAL, logger="__main__")
    caplog.set_level(logging.CRITICAL, logger="combustion")
    caplog.set_level(logging.CRITICAL)


def test_fast_dev_run():
    sys.argv = [sys.argv[0], "trainer=test"]
    runpy.run_module("examples.basic", run_name="__main__", alter_sys=True)


def test_multirun():
    sys.argv = [sys.argv[0], "-m", "trainer=test", "model.params.batch_size=8,32"]
    runpy.run_module("examples.basic", run_name="__main__", alter_sys=True)


def test_multirun_handles_exception():
    sys.argv = [sys.argv[0], "-m", "trainer=test", "model.params.batch_size=-1, 8"]
    runpy.run_module("examples.basic", run_name="__main__", alter_sys=True)


@pytest.mark.parametrize("ex", [KeyboardInterrupt, SystemExit])
def test_multirun_abort(mocker, ex):
    mocker.patch("pytorch_lightning.Trainer.fit", side_effect=ex)
    sys.argv = [sys.argv[0], "-m", "trainer=test", "model.params.batch_size=-1, 8"]
    with pytest.raises(ex):
        runpy.run_module("examples.basic", run_name="__main__", alter_sys=True)


@pytest.mark.skip
def test_lr_auto_find():
    sys.argv = [sys.argv[0], "trainer=test", "trainer.params.auto_lr_find=True", "trainer.params.fast_dev_run=False"]
    runpy.run_module("examples.basic", run_name="__main__", alter_sys=True)


@pytest.mark.skip
def test_dev_run():
    sys.argv = [sys.argv[0], "trainer=test", "trainer.params.fast_dev_run=False"]
    runpy.run_module("examples.basic", run_name="__main__", alter_sys=True)
