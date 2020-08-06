#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from pytorch_lightning.utilities.exceptions import MisconfigurationException

from combustion.lightning import HydraMixin


class Subclass(HydraMixin, pl.LightningModule):
    def __init__(self, config, **hparams):
        super(Subclass, self).__init__()
        self.hparams = hparams
        self.config = config
        self.trainer = pl.Trainer()
        self.l = nn.Linear(10, 1)
        self.trainer.optimizer = torch.optim.Adam(self.l.parameters(), 0.002)
        if "criterion" in hparams.keys():
            self.criterion = hparams["criterion"]
            del hparams["criterion"]

    def forward(self, x):
        return torch.rand_like(x, requires_grad=True)


class TrainOnlyModel(Subclass):
    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y_hat)
        return {"loss": loss}


class TrainAndValidateModel(TrainOnlyModel):
    def validation_step(self, batch, batch_nb):
        return self.training_step(batch, batch_nb)


class TrainTestValidateModel(TrainAndValidateModel):
    def test_step(self, batch, batch_nb):
        return self.training_step(batch, batch_nb)


@pytest.fixture
def hydra():
    hydra = pytest.importorskip("hydra")
    return hydra


@pytest.fixture
def cfg(torch):
    pytest.importorskip("torchvision")
    omegaconf = pytest.importorskip("omegaconf")
    cfg = {
        "optimizer": {"name": "adam", "target": "torch.optim.Adam", "params": {"lr": 0.002,},},
        "model": {
            "target": TrainTestValidateModel.__module__ + ".TrainTestValidateModel",
            "params": {
                "in_features": 10,
                "out_features": 10,
                "batch_size": 32,
                "criterion": {"target": "torch.nn.BCEWithLogitsLoss", "params": {}},
            },
        },
        "schedule": {
            "interval": "step",
            "monitor": "val_loss",
            "frequency": 1,
            "target": "torch.optim.lr_scheduler.OneCycleLR",
            "params": {
                "max_lr": 0.002,
                "epochs": 10,
                "steps_per_epoch": "none",
                "pct_start": 0.05,
                "div_factor": 25.0,
                "final_div_factor": 10000.0,
                "anneal_strategy": "cos",
            },
        },
        "dataset": {
            "stats_sample_size": 100,
            "stats_dim": 0,
            "train": {
                "num_workers": 4,
                "target": "torchvision.datasets.FakeData",
                "params": {
                    "size": 100,
                    "image_size": [1, 64, 64],
                    "transform": {"target": "torchvision.transforms.ToTensor", "params": {}},
                },
            },
            "validate": {
                "num_workers": 4,
                "target": "torchvision.datasets.FakeData",
                "params": {
                    "size": 100,
                    "image_size": [1, 64, 64],
                    "transform": {"target": "torchvision.transforms.ToTensor", "params": {}},
                },
            },
            "test": {
                "num_workers": 4,
                "target": "torchvision.datasets.FakeData",
                "params": {
                    "size": 100,
                    "image_size": [1, 64, 64],
                    "transform": {"target": "torchvision.transforms.ToTensor", "params": {}},
                },
            },
        },
        "trainer": {"target": "pytorch_lightning.Trainer", "params": {"fast_dev_run": True,}},
    }
    return omegaconf.DictConfig(cfg)


@pytest.fixture
def trainer():
    pl = pytest.importorskip("pytorch_lightning")
    return pl.Trainer()


def test_constructor_sets_hparams(cfg):
    hparams = cfg["model"]["params"]
    model = Subclass(cfg, **hparams)
    for k, v in hparams.items():
        assert model.hparams[k] == v


def test_constructor_sets_config(cfg):
    hparams = cfg["model"]["params"]
    model = Subclass(cfg, **hparams)
    assert model.config == cfg


def test_instantiate_with_hydra(cfg, hydra):
    model = hydra.utils.instantiate(cfg.model, cfg)
    assert isinstance(model, Subclass)


def test_instantiate_recursive(hydra):
    cfg = {
        "target": "torch.nn.Sequential",
        "params": [
            {"target": "torch.nn.Linear", "params": {"in_features": 10, "out_features": 10,}},
            {"target": "torch.nn.Linear", "params": {"in_features": 10, "out_features": 10,}},
        ],
    }

    model = HydraMixin.instantiate(cfg)
    assert isinstance(model, torch.nn.Sequential)


@pytest.mark.parametrize(
    ["target", "exception"],
    [
        pytest.param("combustion.nn.NonExistantClass", ModuleNotFoundError),
        pytest.param("torch.nn.NonExistantClass", ModuleNotFoundError),
    ],
)
def test_instantiate_report_error(hydra, target, exception):
    cfg = {"target": target}
    with pytest.raises(exception):
        HydraMixin.instantiate(cfg)


@pytest.mark.parametrize("scheduled", [True, False])
def test_configure_optimizer(torch, cfg, hydra, scheduled):
    hparams = cfg["model"]["params"]
    if not scheduled:
        del cfg["schedule"]
    model = Subclass(cfg, **hparams)
    model.prepare_data()

    if not scheduled:
        optim = model.configure_optimizers()
        assert isinstance(optim, torch.optim.Adam)
    else:
        optims, schedulers = model.configure_optimizers()
        assert isinstance(optims[0], torch.optim.Adam)
        assert isinstance(schedulers[0], dict)
        assert isinstance(schedulers[0]["scheduler"], torch.optim.lr_scheduler.OneCycleLR)


@pytest.mark.parametrize("missing", ["interval", "frequency"])
def test_configure_optimizer_missing_keys(torch, cfg, hydra, missing):
    hparams = cfg["model"]["params"]
    del cfg["schedule"][missing]
    model = Subclass(cfg, **hparams)
    model.prepare_data()

    with pytest.raises(pl.utilities.exceptions.MisconfigurationException):
        optims, schedulers = model.configure_optimizers()


def test_configure_optimizer_warn_no_monitor_key(torch, cfg, hydra):
    hparams = cfg["model"]["params"]
    del cfg["schedule"]["monitor"]
    model = Subclass(cfg, **hparams)
    model.prepare_data()

    with pytest.warns(UserWarning):
        optims, schedulers = model.configure_optimizers()


@pytest.mark.parametrize("scheduled", [True, False])
def test_get_lr(scheduled, cfg, hydra):
    if not scheduled:
        del cfg["schedule"]

    model = hydra.utils.instantiate(cfg.model, cfg)
    model.prepare_data()
    model.configure_optimizers()
    assert model.get_lr() == cfg["optimizer"]["params"]["lr"]


@pytest.mark.parametrize("params", [True, False])
def test_recursive_instantiate(cfg, params):
    cfg["model"]["params"]["test"] = {"target": "torch.nn.BCELoss", "params": {}}

    if params:
        cfg["model"]["params"]["test"]["params"] = {"reduction": "none"}

    model = HydraMixin.instantiate(cfg.model, cfg, foo=2)
    assert isinstance(model.hparams["test"], torch.nn.BCELoss)
    assert model.hparams["foo"] == 2
    assert model.config == cfg


def test_recursive_instantiate_preserves_cfg(cfg):
    key = {"target": "torch.nn.BCELoss", "params": {"reduction": "none"}}
    cfg["model"]["params"]["test"] = key
    model = HydraMixin.instantiate(cfg.model, cfg, foo=2)
    assert "test" in model.config["model"]["params"].keys()
    assert model.config["model"]["params"]["test"] == key


def test_recursive_instantiate_interpolated():
    yaml = r"""
    model:
        in_channels: 4
        out_channels: 8
        kernel_size: 3
        target: torch.nn.Sequential
        params:
            - target: torch.nn.Conv2d
              params:
                in_channels: ${model.in_channels}
                out_channels: ${model.out_channels}
                kernel_size: ${model.kernel_size}
            - target: torch.nn.Conv2d
              params:
                in_channels: ${model.in_channels}
                out_channels: ${model.out_channels}
                kernel_size: ${model.kernel_size}
    """
    cfg = OmegaConf.create(yaml)
    model = HydraMixin.instantiate(cfg.model)
    assert isinstance(model, torch.nn.Module)


@pytest.mark.parametrize("check", ["train_ds", "val_ds", "test_ds"])
def test_prepare_data(cfg, check):
    model = HydraMixin.instantiate(cfg.model, cfg)
    model.prepare_data()
    assert hasattr(model, check)
    assert isinstance(getattr(model, check), torch.utils.data.Dataset)


@pytest.mark.parametrize("force", [True, False])
def test_prepare_data_forced(cfg, force, mocker):
    model = HydraMixin.instantiate(cfg.model, cfg)
    mock = mocker.MagicMock(spec_set=bool, name="train_ds")
    model.train_ds = mock
    model.prepare_data(force=force)
    model.train_ds

    if force:
        assert model.train_ds != mock
    else:
        assert model.train_ds == mock


@pytest.mark.parametrize(
    "missing, present",
    [
        pytest.param(["validate", "test"], ["train"]),
        pytest.param(["test"], ["train", "validate"]),
        pytest.param(["validate", "train"], ["test"]),
    ],
)
def test_prepare_data_missing_items(cfg, missing, present):
    for k in missing:
        del cfg.dataset[k]
    model = HydraMixin.instantiate(cfg.model, cfg)
    model.prepare_data()


@pytest.mark.parametrize("present", [True, False])
def test_train_dataloader(cfg, present):
    if not present:
        del cfg.dataset["train"]
    model = HydraMixin.instantiate(cfg.model, cfg)
    model.prepare_data()
    dataloader = model.train_dataloader()

    if present:
        assert isinstance(dataloader, torch.utils.data.DataLoader)
    else:
        assert dataloader is None


@pytest.mark.parametrize("present", [True, False])
def test_val_dataloader(cfg, present):
    if not present:
        del cfg.dataset["validate"]
    model = HydraMixin.instantiate(cfg.model, cfg)
    model.prepare_data()
    dataloader = model.val_dataloader()

    if present:
        assert isinstance(dataloader, torch.utils.data.DataLoader)
    else:
        assert dataloader is None


@pytest.mark.parametrize("present", [True, False])
def test_test_dataloader(cfg, present):
    if not present:
        del cfg.dataset["test"]
    model = HydraMixin.instantiate(cfg.model, cfg)
    model.prepare_data()
    dataloader = model.test_dataloader()

    if present:
        assert isinstance(dataloader, torch.utils.data.DataLoader)
    else:
        assert dataloader is None


@pytest.mark.parametrize("subset", ["test", "validate"])
@pytest.mark.parametrize("split", [True, False])
def test_dataloader_from_subset(cfg, subset, split):
    if subset == "test":
        if split:
            cfg.dataset["test"] = 10
        del cfg.dataset["validate"]
    else:
        if split:
            cfg.dataset["validate"] = 10
        del cfg.dataset["test"]

    model = HydraMixin.instantiate(cfg.model, cfg)
    model.prepare_data()

    if subset == "test":
        dataloader = model.test_dataloader()
    else:
        dataloader = model.val_dataloader()
    train_dl = model.train_dataloader()

    # TODO should check shuffle=False, but this is hidden in dataloader.sampler
    assert isinstance(dataloader, torch.utils.data.DataLoader)
    for key in ["pin_memory", "batch_size", "num_workers", "drop_last"]:
        assert getattr(dataloader, key) == getattr(train_dl, key)


@pytest.mark.parametrize("dim", [pytest.param(0, id="dim=0"), pytest.param(-3, id="dim=-3"),])
@pytest.mark.parametrize(
    "num_examples",
    [
        pytest.param(100, id="num_examples=100"),
        pytest.param(0, id="num_examples=0"),
        pytest.param("all", id="num_examples=all"),
    ],
)
@pytest.mark.parametrize("index", [pytest.param(0, id="index=0"), pytest.param(-2, id="index=-2"),])
def test_get_train_ds_statistics(cfg, dim, num_examples, index):
    cfg.dataset["stats_sample_size"] = num_examples
    cfg.dataset["stats_dim"] = dim
    cfg.dataset["stats_index"] = index

    model = HydraMixin.instantiate(cfg.model, cfg)
    model.prepare_data()
    num_channels = model.train_ds[0][0].shape[dim]

    for attr in ["channel_mean", "channel_variance", "channel_max", "channel_min"]:
        if isinstance(num_examples, str) or num_examples > 0:
            assert hasattr(model, attr)
            x = getattr(model, attr)
            assert isinstance(x, torch.Tensor)
            assert x.shape[0] == num_channels
        else:
            assert not hasattr(model, attr)


@pytest.mark.parametrize("index", [pytest.param(10), pytest.param(4), pytest.param(-4),])
def test_get_train_ds_statistics_index_error_handling(cfg, index):
    cfg.dataset["stats_index"] = index
    model = HydraMixin.instantiate(cfg.model, cfg)
    with pytest.raises(MisconfigurationException):
        model.prepare_data()


@pytest.mark.parametrize("dim", [pytest.param(10), pytest.param(-10),])
def test_get_train_ds_statistics_dim_error_handling(cfg, dim):
    cfg.dataset["stats_dim"] = dim
    model = HydraMixin.instantiate(cfg.model, cfg)
    with pytest.raises(MisconfigurationException):
        model.prepare_data()


@pytest.mark.parametrize(
    "num_examples", [pytest.param("BAD"), pytest.param(-1),],
)
def test_get_train_ds_statistics_num_examples_error_handling(cfg, num_examples):
    cfg.dataset["stats_sample_size"] = num_examples
    model = HydraMixin.instantiate(cfg.model, cfg)
    with pytest.raises(MisconfigurationException):
        model.prepare_data()


def test_statistics_set_only_once(cfg, mocker):
    cfg.dataset["stats_sample_size"] = 100
    cfg.dataset["stats_dim"] = -3

    model = HydraMixin.instantiate(cfg.model, cfg)
    model.prepare_data()
    old_mean = model.channel_mean

    model.channel_mean *= 0.0
    model.prepare_data()
    new_mean = model.channel_mean

    assert torch.allclose(old_mean, new_mean)


@pytest.mark.parametrize("subset", ["test", "validate"])
def test_dataloader_override_batch_size(cfg, subset):
    model_batch_size = cfg.model["params"]["batch_size"]
    new_batch_size = model_batch_size + 1

    cfg.dataset[subset]["batch_size"] = new_batch_size

    model = HydraMixin.instantiate(cfg.model, cfg)
    model.prepare_data()

    if subset == "test":
        dataloader = model.test_dataloader()
    else:
        dataloader = model.val_dataloader()
    train_dl = model.train_dataloader()

    assert train_dl.batch_size == model_batch_size
    assert dataloader.batch_size == new_batch_size


class TestRuntimeBehavior:
    @pytest.fixture(autouse=True, params=[TrainOnlyModel, TrainAndValidateModel, TrainTestValidateModel])
    def model(self, request, cfg):
        target = request.param.__module__ + "." + request.param.__name__
        cfg.model["target"] = target

    @pytest.fixture
    def trainer(self, cfg):
        return HydraMixin.instantiate(cfg.trainer)

    def test_train_only(self, cfg, trainer):
        for key in ["validate", "test"]:
            if key in cfg.dataset.keys():
                del cfg.dataset[key]

        model = HydraMixin.instantiate(cfg.model, cfg)
        trainer.fit(model)

    def test_train_validate(self, cfg, trainer):
        for key in ["test"]:
            if key in cfg.dataset.keys():
                del cfg.dataset[key]

        model = HydraMixin.instantiate(cfg.model, cfg)
        trainer.fit(model)

    def test_train_validate_test(self, cfg, trainer):
        model = HydraMixin.instantiate(cfg.model, cfg)
        trainer.fit(model)
