#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import pytorch_lightning as pl
import torch
from pytorch_lightning.overrides.data_parallel import LightningDistributedDataParallel, LightningDistributedModule
from pytorch_lightning.trainer.optimizers import TrainerOptimizersMixin
from torch import Tensor
from torch.utils.data import DataLoader

from .decorators import cuda_or_skip


# dummy class to hold mixin methods
class _ProcessOptimizers(TrainerOptimizersMixin):
    pass


def check_overriden(model, method):
    try:
        if isinstance(model, (LightningDistributedDataParallel, LightningDistributedModule)):
            model = model.module
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


def expose_distributed_model(model):
    if isinstance(model, LightningDistributedDataParallel):
        model = model.module
    if isinstance(model, LightningDistributedModule):
        model = model.module
    return model


class LightningModuleTest:
    r"""Base class to automate testing of LightningModules with Pytest.

    The following fixtures should be implemented in the subclass:

        * :func:`model`
        * :func:`data`

    Simple tests are provided for the following lifecycle hooks:
        * :func:`configure_optimizers`
        * :func:`prepare_data` (optional)
        * :func:`train_dataloader`
        * :func:`val_dataloader` (optional)
        * :func:`test_dataloader` (optional)
        * :func:`training_step` - single process and with :class:`torch.nn.parallel.DistributedDataParallel`
        * :func:`validation_step` (optional)
        * :func:`test_step` (optional)
        * :func:`validation_epoch_end` (optional)
        * :func:`test_epoch_end` (optional)

    If the model under test does not implement an optional method, the test will be
    skipped.

    The following mock attributes will be attached to your model as ``PropertyMock``
        * :attr:`logger`
        * :attr:`trainer`

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
    DISTRIBUTED: bool = True

    def init_process_group(self):
        if not torch.distributed.is_initialized():
            torch.cuda.set_device(0)
            torch.distributed.init_process_group(
                backend="nccl", world_size=1, rank=0, init_method="tcp://127.0.0.1:23456"
            )

    def _distributed_model(self, model: pl.LightningModule) -> LightningDistributedDataParallel:
        model = LightningDistributedDataParallel(model, device_ids=[0], find_unused_parameters=True)
        return model

    @pytest.fixture
    def model(self):
        raise pytest.UsageError("Must implement model fixture for LightningModuleTest")

    @pytest.fixture
    def data(self):
        raise pytest.UsageError("Must implement data fixture for LightningModuleTest")

    @pytest.fixture
    def distributed_model(self, model):
        self.init_process_group()
        model = model.cuda()
        return self._distributed_model(model)

    @pytest.fixture
    def prepare_data(self, model):
        model = expose_distributed_model(model)
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
        r"""Tests that ``model.configure_optimizers()`` runs and returns the required
        outputs.
        """
        model = expose_distributed_model(model)
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
        r"""Calls ``model.prepare_data()`` to see if any fatal errors are thrown. No
        tests are performed to assess change of state
        """
        if isinstance(model, (LightningDistributedDataParallel, LightningDistributedModule)):
            model = model.module
        model.prepare_data()

    @pytest.mark.usefixtures("prepare_data")
    def test_train_dataloader(self, model: pl.LightningModule):
        r"""Tests that ``model.train_dataloader()`` runs and returns the required output."""
        model = expose_distributed_model(model)
        dl = model.train_dataloader()
        check_dataloader(dl)
        return dl

    @pytest.mark.usefixtures("prepare_data")
    def test_val_dataloader(self, model: pl.LightningModule):
        r"""Tests that ``model.val_dataloader()`` runs and returns the required output."""
        model = expose_distributed_model(model)
        check_overriden(model, "val_dataloader")
        dl = model.val_dataloader()
        check_dataloader(dl)
        return dl

    @pytest.mark.usefixtures("prepare_data")
    def test_test_dataloader(self, model: pl.LightningModule):
        r"""Tests that ``model.test_dataloader()`` runs and returns the required output."""
        model = expose_distributed_model(model)
        check_overriden(model, "test_dataloader")
        dl = model.test_dataloader()
        check_dataloader(dl)
        return dl

    @cuda_or_skip
    @pytest.mark.usefixtures("prepare_data")
    @pytest.mark.parametrize("training", [True, False])
    def test_forward(self, model: pl.LightningModule, data: torch.Tensor, training: bool):
        r"""Calls ``model.forward()`` and tests that the output is not ``None``.

        Because of the size of some models, this test is only run when a GPU is available.
        """
        if isinstance(model, (LightningDistributedDataParallel, LightningDistributedModule)):
            pytest.skip()

        if torch.cuda.is_available():
            model = model.cuda()
            data = data.cuda()

        if training:
            model.train()
        else:
            model.eval()

        _ = model(data)

        assert _ is not None

    @cuda_or_skip
    @pytest.mark.usefixtures("prepare_data")
    @pytest.mark.parametrize(
        "distributed",
        [
            pytest.param(True, id="distributed"),
            pytest.param(False, id="non-distributed"),
        ],
    )
    def test_training_step(self, model: pl.LightningModule, distributed: bool):
        r"""Runs a training step based on the data returned from ``model.train_dataloader()``.
        Tests that the dictionary returned from ``training_step()`` are as required by PyTorch
        Lightning. A backward pass and optimizer step are also performed using the optimizer
        provided by :func:`LightningModule.configure_optimizers`. By default, training steps
        are tested for distributed and non-distributed models using the
        :class:`torch.nn.parallel.DistributedDataParallel` wrapper. Distributed tests can be disabled
        by setting :attr:`LightningModuleTest.DISTRIBUTED` to ``False``.

        Because of the size of some models, this test is only run when a GPU is available.
        """
        if distributed:
            if not self.DISTRIBUTED:
                pytest.skip("LightningModuleTest.DISTRIBUTED was False, skipping distributed training step")
            self.init_process_group()
            model = self._distributed_model(model.cuda())

        if isinstance(model, (LightningDistributedDataParallel, LightningDistributedModule)):
            dl = expose_distributed_model(model).train_dataloader()
        else:
            dl = model.train_dataloader()

        batch = next(iter(dl))

        if torch.cuda.is_available():
            batch = [x.cuda() for x in batch]
            model = model.cuda()

        model.train()

        # TODO this can't handle multiple optimizers
        if isinstance(model, (LightningDistributedDataParallel, LightningDistributedModule)):
            output = expose_distributed_model(model).training_step(batch, 0)
        else:
            output = model.training_step(batch, 0)

        assert isinstance(output, dict)

        assert "loss" in output.keys(), "loss key is required"
        assert_valid_loss(output["loss"])

        # test loss / backward pass, important for distributed operation
        if isinstance(model, (LightningDistributedDataParallel, LightningDistributedModule)):
            output = expose_distributed_model(model).training_step(batch, 0)
            optimizers, lr_scheduler, frequencies = _ProcessOptimizers().init_optimizers(
                expose_distributed_model(model)
            )
        else:
            optimizers, lr_scheduler, frequencies = _ProcessOptimizers().init_optimizers(model)
        optim = optimizers[0]
        loss = output["loss"]
        optim.zero_grad()
        loss.backward()
        optim.step()

        if "log" in output.keys():
            assert isinstance(output["log"], dict)

        if "progress_bar" in output.keys():
            assert isinstance(output["progress_bar"], dict)

        return batch, output

    @cuda_or_skip
    @pytest.mark.usefixtures("prepare_data")
    def test_validation_step(self, model: pl.LightningModule):
        r"""Runs a validation step based on the data returned from ``model.val_dataloader()``.
        Tests that the dictionary returned from ``validation_step()`` are as required by PyTorch
        Lightning.

        Because of the size of some models, this test is only run when a GPU is available.
        """
        if isinstance(model, (LightningDistributedDataParallel, LightningDistributedModule)):
            pytest.skip()

        check_overriden(model, "val_dataloader")
        check_overriden(model, "validation_step")

        dl = model.val_dataloader()
        # TODO this can't handle multiple optimizers
        batch = next(iter(dl))

        if torch.cuda.is_available():
            batch = [x.cuda() for x in batch]
            model = model.cuda()

        model.eval()
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

    @cuda_or_skip
    @pytest.mark.usefixtures("prepare_data")
    def test_validation_epoch_end(self, model: pl.LightningModule):
        r"""Tests that ``validation_epoch_end()`` runs and outputs a dict as required by PyTorch
        Lightning.

        Because of the size of some models, this test is only run when a GPU is available.
        """
        if isinstance(model, (LightningDistributedDataParallel, LightningDistributedModule)):
            pytest.skip()

        check_overriden(model, "val_dataloader")
        check_overriden(model, "validation_step")
        check_overriden(model, "validation_epoch_end")
        dl = model.val_dataloader()

        if torch.cuda.is_available():
            model = model.cuda()
            model.eval()
            outputs = [model.validation_step([x.cuda() for x in batch], 0) for batch in dl]
        else:
            model.eval()
            outputs = [model.validation_step(batch, 0) for batch in dl]

        result = model.validation_epoch_end(outputs)
        assert isinstance(result, dict)
        return outputs, result

    @cuda_or_skip
    @pytest.mark.usefixtures("prepare_data")
    def test_test_step(self, model: pl.LightningModule):
        r"""Runs a testing step based on the data returned from ``model.test_dataloader()``.
        Tests that the dictionary returned from ``test_step()`` are as required by PyTorch
        Lightning.

        Because of the size of some models, this test is only run when a GPU is available.
        """
        if isinstance(model, (LightningDistributedDataParallel, LightningDistributedModule)):
            pytest.skip()

        check_overriden(model, "test_dataloader")
        check_overriden(model, "test_step")

        dl = model.test_dataloader()
        batch = next(iter(dl))

        if torch.cuda.is_available():
            batch = [x.cuda() for x in batch]
            model = model.cuda()

        # TODO this can't handle multiple optimizers
        model.eval()
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

    @cuda_or_skip
    @pytest.mark.usefixtures("prepare_data")
    def test_test_epoch_end(self, model: pl.LightningModule):
        r"""Tests that ``test_epoch_end()`` runs and outputs a dict as required by PyTorch
        Lightning.

        Because of the size of some models, this test is only run when a GPU is available.
        """
        if isinstance(model, (LightningDistributedDataParallel, LightningDistributedModule)):
            pytest.skip()

        check_overriden(model, "test_dataloader")
        check_overriden(model, "test_step")
        check_overriden(model, "test_epoch_end")
        dl = model.test_dataloader()

        if torch.cuda.is_available():
            model = model.cuda()
            model.eval()
            outputs = [model.test_step([x.cuda() for x in batch], 0) for batch in dl]
        else:
            model.eval()
            outputs = [model.test_step(batch, 0) for batch in dl]

        result = model.test_epoch_end(outputs)
        assert isinstance(result, dict)
        return outputs, result
