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


class FakeModel(HydraMixin):
    def __init__(self, hparams: DictConfig):
        super().__init__()
        in_features = hparams.model.params.in_features
        out_features = hparams.model.params.out_features
        kernel = hparams.model.params.kernel
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
        tensorboard_logs = {"train_loss": loss}

        # sample progress bar override with lr logging
        bar = {"lr": self.get_lr()}

        return {"loss": loss, "log": tensorboard_logs, "progress_bar": bar}

    def validation_step(self, batch, batch_nb):
        # OPTIONAL
        x, y = batch
        y_hat = self(x)
        return {"val_loss": F.cross_entropy(y_hat, y)}

    def validation_epoch_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": tensorboard_logs}

    def test_step(self, batch, batch_nb):
        # OPTIONAL
        x, y = batch
        y_hat = self(x)
        return {"test_loss": F.cross_entropy(y_hat, y)}

    def test_epoch_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        logs = {"test_loss": avg_loss}
        return {"avg_test_loss": avg_loss, "log": logs, "progress_bar": logs}
