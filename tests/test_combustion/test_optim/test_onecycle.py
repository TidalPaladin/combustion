#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import torch.nn as nn
from torch.optim import Adam

from combustion.optim import SuperConvergenceLR


@pytest.fixture
def model():
    return nn.Linear(10, 10)


@pytest.fixture
def optimizer(model):
    return Adam(model.parameters(), lr=0.001)


@pytest.mark.parametrize(
    "epochs,steps_per_epoch",
    [
        pytest.param(10, 100),
        pytest.param(5, 100),
        pytest.param(10, 200),
    ],
)
@pytest.mark.parametrize(
    "pct_warmup,pct_cooldown,div_factor,final_div_factor",
    [
        pytest.param(0.3, 0.3, 4, 100),
        pytest.param(0.3, 0.3, 10, 1000),
        pytest.param(0.25, 0.25, 4, 100),
        pytest.param(0.5, 0.5, 4, 100),
        pytest.param(0.8, 0.5, 4, 100, marks=pytest.mark.xfail(raises=ValueError)),
    ],
)
def test_lr_schedule(model, optimizer, epochs, steps_per_epoch, pct_warmup, pct_cooldown, div_factor, final_div_factor):
    max_lr = 0.001
    schedule = SuperConvergenceLR(
        optimizer,
        max_lr,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_warmup=pct_warmup,
        pct_cooldown=pct_cooldown,
        div_factor=div_factor,
        final_div_factor=final_div_factor,
    )

    total_steps = epochs * steps_per_epoch
    values = []

    for i in range(total_steps):
        schedule.step()
        values.append(schedule.get_last_lr()[0])

    assert abs(values[0] - max_lr / div_factor) <= 1e-5
    assert abs(values[int(pct_warmup * total_steps)] - max_lr) <= 1e-5

    if pct_warmup + pct_cooldown == 1.0:
        assert abs(values[-1] - max_lr / div_factor) <= 1e-5
    else:
        assert abs(values[-1] - max_lr / final_div_factor) <= 1e-5
        assert abs(values[int((pct_warmup + pct_cooldown) * total_steps)] - max_lr / div_factor) <= 1e-5


def test_repr(model, optimizer):
    max_lr = 0.001
    schedule = SuperConvergenceLR(
        optimizer,
        max_lr,
        epochs=10,
        steps_per_epoch=10,
    )
    print(schedule)
