#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

import pytest

from combustion.ignite.handlers import SummaryWriter


@pytest.fixture
def summary():
    return pytest.importorskip("pytorch_model_summary.summary")


@pytest.fixture
def model(torch, device):
    return torch.nn.Linear(10, 1).to(device)


@pytest.fixture(autouse=True)
def batch(torch, state, device):
    batch = torch.rand(10).to(device)
    state.batch = batch
    return batch


def test_repr(tmpdir, model):
    filepath = os.path.join(tmpdir, "summary.txt")
    summary = SummaryWriter(model, filepath=filepath)
    out = repr(summary)
    assert filepath in out


def test_overwrite(model, tmpdir, engine):
    filepath = os.path.join(tmpdir, "summary.txt")
    open(filepath, "a").close()
    summary = SummaryWriter(model, filepath=filepath)
    with pytest.raises(FileExistsError):
        summary(engine)


def test_kwargs(model, tmpdir, engine):
    filepath = os.path.join(tmpdir, "summary.txt")
    SummaryWriter(model, filepath=filepath, show_hierarchical=True)


def test_transform(torch, model, tmpdir, engine, state, device):
    batch = (torch.rand(10).to(device), torch.rand(10).to(device))
    state.batch = batch
    transform = lambda batch: batch[0]
    filepath = os.path.join(tmpdir, "summary.txt")
    summary = SummaryWriter(model, transform=transform, filepath=filepath)
    summary(engine)


def test_print(model, engine, mocker):
    fn = mocker.MagicMock()
    summary = SummaryWriter(model, print_fn=fn)
    summary(engine)
    fn.assert_called_once()


def test_logger(model, engine, mocker):
    fn = mocker.MagicMock()
    summary = SummaryWriter(model, log_fn=fn)
    summary(engine)
    fn.assert_called_once()
