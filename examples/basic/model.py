#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging

import torch.nn as nn
import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch.nn import functional as F
from combustion.lightning.mixins import OptimizerMixin
import combustion
from dataclasses import dataclass
from hydra_configs.torch.optim import AdamConf
from hydra_configs.torch.optim.lr_scheduler import OneCycleLRConf
from hydra_configs.torch.utils.data.dataloader import DataLoaderConf
from typing import Any
from torchvision.datasets import FakeData
from torch.optim import Optimizer
import combustion
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate

log = logging.getLogger(__name__)


@dataclass
class CrossEntropyLossConf:
    _target_: str = 'torch.nn.CrossEntropyLoss'


@dataclass
class FakeModelConf:
    _target_: str = 'examples.basic.model.FakeModel'
    in_features: int = 1
    out_features: int = 10
    kernel: int = 3
    optimizer: Any = AdamConf()
    schedule: Any = None
    criterion: Any = CrossEntropyLossConf()


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

        
        del optimizer.params

        self.criterion = instantiate(criterion)
        self.save_hyperparameters()

    def forward(self, x):
        _ = self.l1(x)
        _ = self.l2(_).squeeze()
        _ = self.l3(_)
        return torch.relu(_)

    def configure_optimizers(self):
        optim = instantiate(self.hparams.optimizer, params=self.parameters())
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

cs = ConfigStore.instance()
cs.store(group="model", name="base_fakemodel", node=FakeModelConf)
