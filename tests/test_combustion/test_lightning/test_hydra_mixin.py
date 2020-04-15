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

    def train_dataloader(self):
        return [1, 2, 3]


@pytest.fixture
def hydra():
    hydra = pytest.importorskip("hydra")
    return hydra


@pytest.fixture
def cfg(torch):
    omegaconf = pytest.importorskip("omegaconf")
    cfg = {
        "optimizer": {
            "name": "adam",
            "cls": "torch.optim.Adam",
            "class": "torch.optim.Adam",
            "params": {"lr": 0.002,},
        },
        "model": {
            "cls": Subclass.__module__ + ".Subclass",
            "class": Subclass.__module__ + ".Subclass",
            "params": {"in_features": 10, "out_features": 10, "batch_size": 32},
            "criterion": {"cls": "torch.nn.BCELoss", "class": "torch.nn.BCELoss"},
        },
        "schedule": {
            "interval": "step",
            "monitor": "val_loss",
            "frequency": 1,
            "class": "torch.optim.lr_scheduler.OneCycleLR",
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
def test_configure_unscheduled_optimizer(torch, cfg, hydra, scheduled):
    hparams = cfg["model"]["params"]
    if not scheduled:
        del cfg["schedule"]
    model = Subclass(cfg, **hparams)

    if not scheduled:
        optim = model.configure_optimizers()
        assert isinstance(optim, torch.optim.Adam)
    else:
        optims, schedulers = model.configure_optimizers()
        assert isinstance(optims[0], torch.optim.Adam)
        assert isinstance(schedulers[0], dict)
        assert isinstance(schedulers[0]["scheduler"], torch.optim.lr_scheduler.OneCycleLR)


@pytest.mark.parametrize("scheduled", [True, False])
def test_get_lr(scheduled, cfg, hydra):
    if not scheduled:
        del cfg["schedule"]

    model = hydra.utils.instantiate(cfg.model, cfg)
    model.configure_optimizers()
    assert model.get_lr() == cfg["optimizer"]["params"]["lr"]
