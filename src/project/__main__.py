#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import FakeData

from combustion.lightning import HydraMixin


log = logging.getLogger(__name__)


# define the model
class MNISTModel(HydraMixin, pl.LightningModule):
    def __init__(self, cfg, **hparams):
        super(MNISTModel, self).__init__()
        self.cfg = cfg
        self._hparams = DictConfig(hparams)
        self.hparams = hparams
        self.config = cfg

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
        # y = torch.tensor([y], dtype=torch.long).type_as(y)
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

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
        # y = torch.tensor([y], dtype=torch.long).type_as(y)
        y_hat = self(x)
        return {"test_loss": F.cross_entropy(y_hat, y)}

    def test_epoch_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        logs = {"test_loss": avg_loss}
        return {"avg_test_loss": avg_loss, "log": logs, "progress_bar": logs}

    def train_dataloader(self):
        # REQUIRED
        return DataLoader(
            FakeData(size=10000, image_size=(1, 28, 28), transform=transforms.ToTensor(),),
            num_workers=4,
            batch_size=self._hparams.batch_size,
        )

    def val_dataloader(self):
        # OPTIONAL
        return DataLoader(
            FakeData(size=1000, image_size=(1, 28, 28), transform=transforms.ToTensor(),),
            num_workers=4,
            batch_size=self._hparams.batch_size,
        )

    def test_dataloader(self):
        # OPTIONAL
        return DataLoader(
            FakeData(size=1000, image_size=(1, 28, 28), transform=transforms.ToTensor(),),
            batch_size=self._hparams.batch_size,
            num_workers=4,
        )


# accepts options from the yaml structure in ./conf
# see hydra docs: https://hydra.cc/docs/intro
@hydra.main(config_path="../../conf/config.yaml")
def main(cfg):
    log.info("Initializing")
    log.info("Configuration: \n%s", cfg.pretty())

    # instantiate model (and optimizer) selected in yaml
    # see pytorch lightning docs: https://pytorch-lightning.rtfd.io/en/latest
    model: pl.LightningModule = hydra.utils.instantiate(cfg.model, cfg)

    # instantiate trainer with params as selected in yaml
    # handles tensorboard, checkpointing, etc
    trainer: pl.Trainer = hydra.utils.instantiate(cfg.trainer)

    # train
    log.info("Starting training")
    trainer.fit(model)

    # test
    log.info("Starting testing")
    trainer.test(model)

    log.info("Finished!")


if __name__ == "__main__":
    main()
