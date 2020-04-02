#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest

from combustion.data.dataset import MBBatch


@pytest.fixture
def data(batch):
    return [batch] * 12


@pytest.fixture
def batch(mocker, torch):
    m = mocker.MagicMock(spec_set=MBBatch, name="batch")
    m.frames = torch.rand(10, 1, 3, 5, 5, requires_grad=True)
    m.labels = torch.rand(10, 1, 5, 5, requires_grad=True)
    m.to.return_value = m
    return m


@pytest.fixture
def criterion(torch):
    return torch.nn.MSELoss()


@pytest.fixture
def trainer(ignite, criterion, model, optimizer):
    def process_fn(engine, batch):
        model.train()
        optimizer.zero_grad()
        inputs, labels = batch.frames, batch.labels
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(-3), labels)
        loss.backward()
        optimizer.step()
        return loss.item()

    return ignite.engine.Engine(process_fn)


@pytest.fixture
def model(torch, batch):
    depth_kernel = batch.frames.shape[-3]
    return torch.nn.Conv3d(1, 1, (depth_kernel, 3, 3), padding=(0, 1, 1))


@pytest.fixture
def mock_model(mocker, model, batch):
    m = mocker.MagicMock(spec_set=model, name="model")
    m.return_value = batch.labels.unsqueeze(-3)
    return m


@pytest.fixture
def optimizer(model, torch):
    return torch.optim.Adam(model.parameters())
