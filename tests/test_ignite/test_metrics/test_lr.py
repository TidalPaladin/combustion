#!/usr/bin/env python
# -*- coding: utf-8 -*-


import pytest

from combustion.ignite.metrics import ScheduledLR


@pytest.fixture
def scheduler(mocker, ignite, torch):
    model = torch.nn.Linear(10, 1)
    optimizer = torch.optim.Adam(model.parameters())
    schedule = torch.optim.lr_scheduler.StepLR(optimizer, 10)
    scheduler = ignite.contrib.handlers.param_scheduler.LRScheduler(schedule)
    return scheduler


def test_step(scheduler, engine):
    metric = ScheduledLR(scheduler)
    metric.compute()
    metric.completed(engine, "lr")
    x = engine.state.metrics["lr"]
    assert x == 0.001


def test_attach(ignite, scheduler, engine):
    metric = ScheduledLR(scheduler)
    engine = ignite.engine.Engine(lambda x, y: x)
    metric.attach(engine)
