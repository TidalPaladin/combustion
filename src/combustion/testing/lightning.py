#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.utils.data import DataLoader


def check_overriden(model, method):
    try:
        model_method = getattr(model.__class__, method)
        lightning_method = getattr(pl.LightningModule, method)
        if model_method == lightning_method:
            pytest.skip(f"Method {method} not overriden in {model.__class__.__name__}")

    except AttributeError:
        raise pytest.UsageError(f"Error checking for override of method {method}")


def check_dataloader(dl):
    assert isinstance(dl, DataLoader)
    batch = next(iter(dl))
    for example in batch:
        is_tensor = isinstance(example, Tensor)
        is_tensor_tuple = isinstance(example, tuple) and all([isinstance(x, Tensor) for x in example])
        assert is_tensor or is_tensor_tuple


def assert_valid_loss(x):
    assert isinstance(x, Tensor)
    assert x.ndim == 0
    assert x.requires_grad


class LightningModuleTest:
    r"""Base class to automate testing of LightningModules with Pytest.

    The following fixtures should be implemented in the subclass:

        * A `model` fixture which should create and return an instance
        of the model to be tested.

        * A `data` fixture that will be used to test `model.forward()`.
        It should return an input that will be passed to the `forward()` call of
        the model being tested.

    Simple tests are provided for the following lifecycle hooks:
        * `configure_optimizers`
        * `prepare_data` (optional)
        * `train_dataloader`
        * `val_dataloader` (optional)
        * `test_dataloader` (optional)
        * `training_step`
        * `validation_step` (optional)
        * `test_step` (optional)
        * `validation_epoch_end` (optional)
        * `test_epoch_end` (optional)

    If the model under test does not implement an optional method, the test will be
    skipped.

    The following mock attributes will be attached to your model as `PropertyMock`
        * `logger`
        * `trainer`

    Example Usage::
        >>> # minimal example
        >>> class TestModel(LightningModuleTest):
        >>>     @pytest.fixture
        >>>     def model():
        >>>         return ... # return your model here
        >>>
        >>>     @pytest.fixture
        >>>     def data():
        >>>         return torch.rand(2, 1, 10, 10) # will be passed to model.forward()
    """

    @pytest.fixture
    def model(self):
        raise pytest.UsageError("Must implement model fixture for LightningModuleTest")

    @pytest.fixture
    def data(self):
        raise pytest.UsageError("Must implement data fixture for LightningModuleTest")

    @pytest.fixture
    def prepare_data(self, model):
        try:
            model.prepare_data()
        except NotImplementedError:
            pass

    @pytest.fixture(autouse=True)
    def logger(self, model, mocker):
        m = mocker.PropertyMock()
        type(model).logger = m
        return m

    @pytest.fixture(autouse=True)
    def trainer(self, model, mocker):
        m = mocker.PropertyMock(spec_set=pl.Trainer)
        type(model).trainer = m
        return m

    def test_configure_optimizers(self, model: pl.LightningModule):
        optim = model.configure_optimizers()
        is_optimizer = isinstance(optim, torch.optim.Optimizer)
        is_optim_schedule_tuple = (
            isinstance(optim, tuple)
            and len(optim) == 2
            and isinstance(optim[0], list)
            and all([isinstance(x, torch.optim.Optimizer) for x in optim[0]])
            and isinstance(optim[1], list)
            and all([isinstance(x, torch.optim.lr_scheduler._LRScheduler) for x in optim[0]])
        )
        assert is_optimizer or is_optim_schedule_tuple
        return optim

    def test_prepare_data(self, model: pl.LightningModule):
        model.prepare_data()

    @pytest.mark.usefixtures("prepare_data")
    def test_train_dataloader(self, model: pl.LightningModule):
        dl = model.train_dataloader()
        check_dataloader(dl)
        return dl

    @pytest.mark.usefixtures("prepare_data")
    def test_val_dataloader(self, model: pl.LightningModule):
        check_overriden(model, "val_dataloader")
        dl = model.val_dataloader()
        check_dataloader(dl)
        return dl

    @pytest.mark.usefixtures("prepare_data")
    def test_test_dataloader(self, model: pl.LightningModule):
        check_overriden(model, "test_dataloader")
        dl = model.test_dataloader()
        check_dataloader(dl)
        return dl

    @pytest.mark.usefixtures("prepare_data")
    def test_forward(self, model: pl.LightningModule, data: torch.Tensor):
        _ = model(data)
        assert _ is not None

    @pytest.mark.usefixtures("prepare_data")
    def test_training_step(self, model: pl.LightningModule):
        dl = model.train_dataloader()
        batch = next(iter(dl))
        # TODO this can't handle multiple optimizers
        output = model.training_step(batch, 0)
        assert isinstance(output, dict)

        assert "loss" in output.keys(), "loss key is required"
        assert_valid_loss(output["loss"])

        if "log" in output.keys():
            assert isinstance(output["log"], dict)

        if "progress_bar" in output.keys():
            assert isinstance(output["progress_bar"], dict)

        return batch, output

    @pytest.mark.usefixtures("prepare_data")
    def test_validation_step(self, model: pl.LightningModule):
        check_overriden(model, "val_dataloader")
        check_overriden(model, "validation_step")

        dl = model.val_dataloader()
        # TODO this can't handle multiple optimizers
        batch = next(iter(dl))
        output = model.validation_step(batch, 0)

        assert isinstance(output, dict)

        if "loss" in output.keys():
            assert isinstance(output["loss"], Tensor)
            assert output["loss"].shape == (1,)

        if "log" in output.keys():
            assert isinstance(output["log"], dict)

        if "progress_bar" in output.keys():
            assert isinstance(output["progress_bar"], dict)

        return batch, output

    @pytest.mark.usefixtures("prepare_data")
    def test_validation_epoch_end(self, model: pl.LightningModule):
        check_overriden(model, "val_dataloader")
        check_overriden(model, "validation_step")
        check_overriden(model, "validation_epoch_end")
        dl = model.val_dataloader()
        outputs = [model.validation_step(batch, 0) for batch in dl]
        result = model.validation_epoch_end(outputs)
        assert isinstance(result, dict)
        return outputs, result

    @pytest.mark.usefixtures("prepare_data")
    def test_test_step(self, model: pl.LightningModule):
        check_overriden(model, "test_dataloader")
        check_overriden(model, "test_step")

        dl = model.test_dataloader()
        batch = next(iter(dl))
        # TODO this can't handle multiple optimizers
        output = model.test_step(batch, 0)

        assert isinstance(output, dict)

        if "loss" in output.keys():
            assert isinstance(output["loss"], Tensor)
            assert output["loss"].shape == (1,)

        if "log" in output.keys():
            assert isinstance(output["log"], dict)

        if "progress_bar" in output.keys():
            assert isinstance(output["progress_bar"], dict)
        return batch, output

    @pytest.mark.usefixtures("prepare_data")
    def test_test_epoch_end(self, model: pl.LightningModule):
        check_overriden(model, "test_dataloader")
        check_overriden(model, "test_step")
        check_overriden(model, "test_epoch_end")
        dl = model.test_dataloader()
        outputs = [model.test_step(batch, 0) for batch in dl]
        result = model.test_epoch_end(outputs)
        assert isinstance(result, dict)
        return outputs, result
