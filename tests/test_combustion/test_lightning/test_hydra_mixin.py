#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import pytorch_lightning as pl
import torch
import torch.nn as nn

from combustion.lightning import HydraMixin


class Subclass(HydraMixin, pl.LightningModule):
    def __init__(self, config, **hparams):
        super(Subclass, self).__init__()
        self.hparams = hparams
        self.config = config
        self.trainer = pl.Trainer()
        self.l = nn.Linear(10, 1)
        self.trainer.optimizer = torch.optim.Adam(self.l.parameters(), 0.002)

    def forward(self, x):
        return x


@pytest.fixture
def hydra():
    hydra = pytest.importorskip("hydra")
    return hydra


@pytest.fixture
def cfg(torch):
    omegaconf = pytest.importorskip("omegaconf")
    cfg = {
        "optimizer": {"name": "adam", "cls": "torch.optim.Adam", "params": {"lr": 0.002,},},
        "model": {
            "cls": Subclass.__module__ + ".Subclass",
            "params": {"in_features": 10, "out_features": 10, "batch_size": 32},
            "criterion": {"cls": "torch.nn.BCELoss"},
        },
        "schedule": {
            "interval": "step",
            "monitor": "val_loss",
            "frequency": 1,
            "cls": "torch.optim.lr_scheduler.OneCycleLR",
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
            "num_workers": 4,
            "train": {
                "cls": "torchvision.datasets.FakeData",
                "params": {
                    "size": 100,
                    "image_size": [1, 64, 64],
                    "transform": {"cls": "torchvision.transforms.ToTensor",},
                },
            },
            "validate": {
                "cls": "torchvision.datasets.FakeData",
                "params": {
                    "size": 100,
                    "image_size": [1, 64, 64],
                    "transform": {"cls": "torchvision.transforms.ToTensor",},
                },
            },
            "test": {
                "cls": "torchvision.datasets.FakeData",
                "params": {
                    "size": 100,
                    "image_size": [1, 64, 64],
                    "transform": {"cls": "torchvision.transforms.ToTensor",},
                },
            },
        },
    }
    return omegaconf.DictConfig(cfg)


@pytest.fixture
def trainer():
    pl = pytest.importorskip("pytorch_lightning")
    return pl.Trainer()


def test_constructor_sets_hparams(cfg):
    hparams = cfg["model"]["params"]
    model = Subclass(cfg, **hparams)
    assert model.hparams == hparams


def test_constructor_sets_config(cfg):
    hparams = cfg["model"]["params"]
    model = Subclass(cfg, **hparams)
    assert model.config == cfg


def test_instantiate_with_hydra(cfg, hydra):
    model = hydra.utils.instantiate(cfg.model, cfg)
    assert isinstance(model, Subclass)


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
    cfg["model"]["params"]["test"] = {
        "cls": "torch.nn.BCELoss",
    }

    if params:
        cfg["model"]["params"]["test"]["params"] = {"reduction": "none"}

    model = HydraMixin.instantiate(cfg.model, cfg, foo=2)
    assert isinstance(model.hparams["test"], torch.nn.BCELoss)
    assert model.hparams["foo"] == 2
    assert model.config == cfg


def test_recursive_instantiate_preserves_cfg(cfg):
    key = {"cls": "torch.nn.BCELoss", "params": {"reduction": "none"}}
    cfg["model"]["params"]["test"] = key
    model = HydraMixin.instantiate(cfg.model, cfg, foo=2)
    assert "test" in model.config["model"]["params"].keys()
    assert model.config["model"]["params"]["test"] == key


@pytest.mark.parametrize("check", ["train_ds", "val_ds", "test_ds"])
def test_prepare_data(cfg, check):
    model = HydraMixin.instantiate(cfg.model, cfg)
    model.prepare_data()
    assert hasattr(model, check)
    assert isinstance(getattr(model, check), torch.utils.data.Dataset)


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
