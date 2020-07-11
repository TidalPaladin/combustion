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
    def __init__(self, cfg, **hparams):
        super(FakeModel, self).__init__()
        self.config = cfg
        self._hparams = DictConfig(hparams)
        self.hparams = hparams

        self.l1 = torch.nn.Conv2d(self._hparams.in_features, self._hparams.out_features, self._hparams.kernel)
        self.l2 = torch.nn.AdaptiveAvgPool2d(1)
        self.l3 = torch.nn.Linear(self._hparams.out_features, 10)

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


combustion.initialize(config_path="../../examples/basic/conf", config_name="config")


@hydra.main(config_path="./conf", config_name="config")
def main(cfg):
    return combustion.main(cfg)


if __name__ == "__main__":
    main()
    combustion.check_exceptions()
