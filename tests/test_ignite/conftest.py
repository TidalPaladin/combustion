#!/usr/bin/env python
# -*- coding: utf-8 -*-


import pytest


@pytest.fixture(params=[True, False])
def device(torch, request):
    if request.param and not torch.cuda.is_available():
        pytest.skip()
    if request.param:
        return torch.device("cuda")
    else:
        return torch.device("cpu")


@pytest.fixture
def state(ignite, torch, mocker):
    state = mocker.MagicMock(name="state")
    state.epoch = 1
    state.batch = (torch.rand(2, 1, 10, 10), torch.rand(2, 1, 10, 10))
    state.iteration = 1
    state.metrics = {"mse": 0.1}
    return state


@pytest.fixture
def engine(state, mocker):
    engine = mocker.MagicMock(name="engine")
    engine.state = state
    return engine
