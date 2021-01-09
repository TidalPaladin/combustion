#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch.nn import functional as F

from combustion.lightning import HydraMixin
import combustion


log = logging.getLogger(__name__)


# define the model
class FakeModel(HydraMixin, pl.LightningModule):
    def __init__(self, in_features, out_features, kernel):
        super().__init__()
        self.l1 = torch.nn.Conv2d(in_features, out_features, kernel)
        self.l2 = torch.nn.AdaptiveAvgPool2d(1)
        self.l3 = torch.nn.Linear(out_features, 10)
        self.save_hyperparameters()

    def forward(self, x):
        _ = self.l1(x)
        _ = self.l2(_).squeeze()
        _ = self.l3(_)
        return torch.relu(_)

    def training_step(self, batch, batch_nb):
        # REQUIRED
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        # sample progress bar override with lr logging
        self.log("train/lr", self.get_lr(), on_step=True, prog_bar=True)
        self.log("train/loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_nb):
        # OPTIONAL
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("val/loss", loss)
        return loss

    def test_step(self, batch, batch_nb):
        # OPTIONAL
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("test/loss", loss)
        return loss


combustion.initialize(config_path="./conf", config_name="config")


@hydra.main(config_path="./conf", config_name="config")
def main(cfg):
    return combustion.main(cfg)


if __name__ == "__main__":
    main()
    combustion.check_exceptions()
