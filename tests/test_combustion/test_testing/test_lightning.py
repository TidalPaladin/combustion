#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gc

import pytest
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

from combustion.testing import LightningModuleTest


class Dataset(data.Dataset):
    def __len__(self):
        return 10

    def __getitem__(self, pos):
        x = torch.rand(1, 10, 10).float()
        x.requires_grad = True
        y = torch.rand(10, 10, 10).float()
        return x, y


class MinimalModel(pl.LightningModule):
    def __init__(self, in_features, out_features, kernel):
        super(MinimalModel, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel = kernel
        self.conv = nn.Conv2d(in_features, out_features, kernel, padding=(kernel // 2))

    def forward(self, x):
        return self.conv(x)

    def prepare_data(self):
        self.train_ds = Dataset()

    def train_dataloader(self):
        return data.DataLoader(self.train_ds, 2)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), 0.001)

    def criterion(self, inputs, targets):
        return F.binary_cross_entropy_with_logits(inputs, targets)

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}


class Model(MinimalModel):
    def prepare_data(self):
        self.train_ds = Dataset()
        self.val_ds = Dataset()
        self.test_ds = Dataset()

    def val_dataloader(self):
        return data.DataLoader(self.val_ds, 2)

    def test_dataloader(self):
        return data.DataLoader(self.test_ds, 2)

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        return {"val_loss": self.criterion(y_hat, y)}

    def validation_epoch_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        step = self.trainer.global_step
        self.logger.experiment.add_image("img", torch.rand(1, 10, 10), step)
        return {"avg_val_loss": avg_loss, "log": tensorboard_logs}

    def test_step(self, batch, batch_nb):
        # OPTIONAL
        x, y = batch
        y_hat = self(x)
        return {"test_loss": self.criterion(y_hat, y)}

    def test_epoch_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        logs = {"test_loss": avg_loss}
        return {"avg_test_loss": avg_loss, "log": logs, "progress_bar": logs}


class TestLightningModuleTest(LightningModuleTest):
    DISTRIBUTED = False

    @pytest.fixture
    def model(self):
        yield Model(1, 10, 3)
        gc.collect()

    @pytest.fixture
    def data(self):
        return torch.rand(2, 1, 10, 10)


class TestMinimalLightningModuleTest(LightningModuleTest):
    DISTRIBUTED = False

    @pytest.fixture
    def model(self):
        yield MinimalModel(1, 10, 3)
        gc.collect()

    @pytest.fixture
    def data(self):
        return torch.rand(2, 1, 10, 10)


class TestDistributedLightningModuleTest(LightningModuleTest):
    @pytest.fixture(params=[True, False])
    def model(self, request):
        yield Model(1, 10, 3)
        gc.collect()

    @pytest.fixture
    def data(self):
        return torch.rand(2, 1, 10, 10)
