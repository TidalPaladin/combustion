#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
import runpy
import sys

import pytest
import pytorch_lightning as pl

from combustion import MultiRunError


pytest.importorskip("torchvision", reason="test requires torchvision")


@pytest.mark.filterwarnings("ignore: .*To copy construct from a tensor, it is recommended to use.*")
@pytest.fixture(autouse=True)
def ignore_warning():
    ...


@pytest.fixture(autouse=False)
def set_caplog(caplog):
    caplog.set_level(logging.CRITICAL, logger="__main__")
    caplog.set_level(logging.CRITICAL, logger="combustion")
    caplog.set_level(logging.CRITICAL)


@pytest.mark.ci_skip
@pytest.mark.parametrize("deterministic", [True, False])
@pytest.mark.filterwarnings("ignore: .*To copy construct from a tensor, it is recommended to use.*")
def test_fast_dev_run(mocker, deterministic):
    pl.Trainer()
    fit = mocker.spy(pl.Trainer, "fit")
    test = mocker.spy(pl.Trainer, "test")
    sys.argv = [
        sys.argv[0],
        "trainer=test",
        "trainer.catch_exceptions=False",
        f"trainer.params.deterministic={deterministic}",
    ]
    runpy.run_module("examples.basic", run_name="__main__", alter_sys=True)
    fit.assert_called()
    test.assert_called_once()


@pytest.mark.ci_skip
def test_skip_train(mocker):
    pl.Trainer()
    fit = mocker.spy(pl.Trainer, "fit")
    test = mocker.spy(pl.Trainer, "test")
    sys.argv = [
        sys.argv[0],
        "trainer=test",
        "trainer.catch_exceptions=False",
        "trainer.test_only=True",
    ]
    runpy.run_module("examples.basic", run_name="__main__", alter_sys=True)
    fit.assert_not_called()
    test.assert_called_once()


@pytest.mark.ci_skip
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


def test_preprocess_train(tmp_path):
    pytest.importorskip("torchvision")
    size = 100
    sys.argv = [
        sys.argv[0],
        "trainer=test",
        "trainer.catch_exceptions=False",
        f"dataset.train.params.size={size}",
        f"trainer.preprocess_train_path={tmp_path}",
    ]
    runpy.run_module("examples.basic", run_name="__main__", alter_sys=True)
    num_files_written = len(os.listdir(tmp_path))
    assert num_files_written >= size


@pytest.mark.ci_skip
@pytest.mark.filterwarnings("ignore: .*To copy construct from a tensor, it is recommended to use.*")
def test_load_checkpoint(tmp_path):
    sys.argv = [
        sys.argv[0],
        "trainer=test",
        "trainer.catch_exceptions=False",
        "trainer.params.max_epochs=2",
        "trainer.params.fast_dev_run=false",
        f"trainer.params.default_root_dir={tmp_path}",
    ]

    runpy.run_module("examples.basic", run_name="__main__", alter_sys=True)

    # rename checkpoint file, = in epoch=X.ckpt breaks hydra
    checkpoint = os.path.join(tmp_path, "epoch=1-step=19.ckpt")
    dest = os.path.join(tmp_path, "epoch1.ckpt")
    os.rename(checkpoint, dest)
    checkpoint = dest

    sys.argv.append(f"trainer.load_from_checkpoint={checkpoint}")
    runpy.run_module("examples.basic", run_name="__main__", alter_sys=True)


@pytest.mark.ci_skip
def test_initialize_checks_hydra_version(mocker):
    mocker.patch("hydra.__version__", new="1.0.0rc1")
    sys.argv = [sys.argv[0], "-m", "trainer=test", "model.params.in_channels=8,32"]
    with pytest.raises(ImportError):
        runpy.run_module("examples.basic", run_name="__main__", alter_sys=True)


# NOTE: for some reason, this test needs to run before the multirun tests


@pytest.mark.ci_skip
@pytest.mark.filterwarnings("ignore: .*To copy construct from a tensor, it is recommended to use.*")
def test_lr_auto_find():
    sys.argv = [
        sys.argv[0],
        "trainer=test",
        "trainer.params.auto_lr_find=True",
        "trainer.params.fast_dev_run=False",
        "dataset.batch_size=8",
    ]
    runpy.run_module("examples.basic", run_name="__main__", alter_sys=True)


@pytest.mark.ci_skip
@pytest.mark.filterwarnings("ignore: .*To copy construct from a tensor, it is recommended to use.*")
def test_multirun():
    sys.argv = [sys.argv[0], "-m", "trainer=test", "dataset.batch_size=8,32"]
    runpy.run_module("examples.basic", run_name="__main__", alter_sys=True)


@pytest.mark.ci_skip
@pytest.mark.filterwarnings("ignore: .*To copy construct from a tensor, it is recommended to use.*")
def test_multirun_from_yaml():
    sys.argv = [sys.argv[0], "-m", "trainer=test", "sweeper=sweep1"]
    runpy.run_module("examples.basic", run_name="__main__", alter_sys=True)


@pytest.mark.ci_skip
@pytest.mark.filterwarnings("ignore: .*To copy construct from a tensor, it is recommended to use.*")
def test_multirun_handles_exception():
    sys.argv = [sys.argv[0], "-m", "trainer=test", "dataset.batch_size=-1, 8"]
    with pytest.raises(MultiRunError):
        runpy.run_module("examples.basic", run_name="__main__", alter_sys=True)


@pytest.mark.ci_skip
@pytest.mark.parametrize("ex", [KeyboardInterrupt, SystemExit])
def test_multirun_abort(mocker, ex):
    mocker.patch("pytorch_lightning.Trainer.fit", side_effect=ex)
    sys.argv = [sys.argv[0], "-m", "trainer=test", "dataset.batch_size=-1, 8"]
    with pytest.raises(ex):
        runpy.run_module("examples.basic", run_name="__main__", alter_sys=True)


@pytest.mark.ci_skip
@pytest.mark.skip
def test_dev_run():
    sys.argv = [sys.argv[0], "trainer=test", "trainer.params.fast_dev_run=False"]
    runpy.run_module("examples.basic", run_name="__main__", alter_sys=True)
