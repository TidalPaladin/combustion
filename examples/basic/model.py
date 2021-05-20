#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging

import torch.nn as nn
import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, MISSING
from torch.nn import functional as F
from combustion.lightning.mixins import OptimizerMixin
import combustion
from dataclasses import dataclass
from hydra_configs.torch.optim import AdamConf
from hydra_configs.torch.optim.lr_scheduler import OneCycleLRConf
from hydra_configs.torch.utils.data.dataloader import DataLoaderConf
from typing import Any, Optional
from torchvision.datasets import FakeData
from torch.optim import Optimizer
import combustion
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from combustion.util import dataclass_init, hydra_dataclass, Instantiable

log = logging.getLogger(__name__)


@hydra_dataclass(spec=torch.nn.CrossEntropyLoss, name="celoss", group="model/criterion")
class CrossEntropyLossConf:
    ...


@hydra_dataclass(target='FakeModel', name="fakemodel", group="model")
class FakeModelConf:
    in_features: int = 1
    out_features: int = 10
    kernel: int = 3
    criterion: Any = CrossEntropyLossConf()
    optimizer: Any = AdamConf()
    schedule: Any = None


class FakeModel(pl.LightningModule, OptimizerMixin):
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        kernel: int, 
        optimizer: Any, 
        schedule: Any,
        criterion: Any
    ):
        super().__init__()
        self.l1 = torch.nn.Conv2d(in_features, out_features, kernel)
        self.l2 = torch.nn.AdaptiveAvgPool2d(1)
        self.l3 = torch.nn.Linear(out_features, 10)

        self.criterion = instantiate(criterion)
        self.save_hyperparameters()
        self._optim = optimizer

    def forward(self, x):
        _ = self.l1(x)
        _ = self.l2(_).squeeze()
        _ = self.l3(_)
        return torch.relu(_)

    def configure_optimizers(self):
        optim = instantiate(self._optim, params=self.parameters())
        return optim

    def training_step(self, batch, batch_nb):
        # REQUIRED
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def validation_step(self, batch, batch_nb):
        # OPTIONAL
        x, y = batch
        y_hat = self(x)
        return F.cross_entropy(y_hat, y)

    def test_step(self, batch, batch_nb):
        # OPTIONAL
        x, y = batch
        y_hat = self(x)
        return F.cross_entropy(y_hat, y)
