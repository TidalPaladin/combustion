#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging

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
from torchvision.transforms import ToTensor
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
import combustion
from combustion.util import dataclass_init, hydra_dataclass


@hydra_dataclass(target="FakeDataModule", name="fakedata", group="data", recursive=False)
class FakeDataModuleConf:
    train_dl: DataLoaderConf = DataLoaderConf(pin_memory=True, batch_size=8, shuffle=True)
    val_dl: DataLoaderConf = DataLoaderConf(batch_size=4, shuffle=False)
    test_dl: DataLoaderConf = DataLoaderConf(batch_size=4, shuffle=False)
    predict_dl: DataLoaderConf = DataLoaderConf(batch_size=4, shuffle=False)

class FakeDataModule(pl.LightningDataModule):
    def __init__(
        self, 
        train_dl: DataLoaderConf, 
        val_dl: DataLoaderConf, 
        test_dl: DataLoaderConf,
        predict_dl: DataLoaderConf
    ):
        super().__init__()
        self.train_ds = FakeData(size=10000, image_size=(1, 128, 128), transform=ToTensor())
        self.val_ds = FakeData(size=100, image_size=(1, 128, 128), transform=ToTensor())
        self.test_ds = FakeData(size=100, image_size=(1, 128, 128), transform=ToTensor())
        self.predict_ds = FakeData(size=100, image_size=(1, 128, 128), transform=ToTensor())
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.test_dl = test_dl
        self.predict_dl = predict_dl

    def train_dataloader(self):
        return instantiate(self.train_dl, dataset=self.train_ds)

    def val_dataloader(self):
        return instantiate(self.val_dl, dataset=self.val_ds)

    def test_dataloader(self):
        return instantiate(self.test_dl, dataset=self.test_ds)

    def predict_dataloader(self):
        return instantiate(self.predict_dl, dataset=self.predict_ds)
