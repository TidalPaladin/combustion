#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import torch

from combustion.testing import (
    assert_has_gradient,
    assert_in_eval_mode,
    assert_in_training_mode,
    assert_is_int_tensor,
    assert_tensors_close,
    assert_zero_grad,
)


@pytest.mark.parametrize("has_grad", [True, False])
def test_assert_has_gradient(has_grad):
    x = torch.rand(10, requires_grad=has_grad)
    module = torch.nn.Linear(10, 10)
    scalar = module(x).sum()
    if not has_grad:
        with pytest.raises(AssertionError):
            assert_has_gradient(module)
    else:
        scalar.backward()
        assert_has_gradient(x)


@pytest.mark.parametrize("training", [True, False])
def test_assert_in_eval_mode(training):
    x = torch.nn.Linear(10, 10)
    if training:
        x.train()
    else:
        x.eval()

    if training:
        with pytest.raises(AssertionError):
            assert_in_eval_mode(x)
    else:
        assert_in_eval_mode(x)


@pytest.mark.parametrize("training", [True, False])
def test_assert_in_training_mode(training):
    x = torch.nn.Linear(10, 10)
    if training:
        x.train()
    else:
        x.eval()

    if not training:
        with pytest.raises(AssertionError):
            assert_in_training_mode(x)
    else:
        assert_in_training_mode(x)


@pytest.mark.parametrize("is_int", [True, False])
def test_assert_is_int_tensor(is_int):
    x = torch.rand(10, 10)
    if is_int:
        x = x.round()

    if not is_int:
        with pytest.raises(AssertionError):
            assert_is_int_tensor(x)
    else:
        assert_is_int_tensor(x)


def test_assert_tensors_close():
    x = torch.rand(10, 10)
    x_clone = x.clone()
    y = torch.rand_like(x)

    assert_tensors_close(x, x_clone)
    with pytest.raises(AssertionError):
        assert_tensors_close(x, y)


@pytest.mark.parametrize("zeroed", [True, False])
def test_zero_grad(zeroed):
    x = torch.rand(10, requires_grad=True)
    module = torch.nn.Linear(10, 10)
    scalar = module(x).sum()

    if not zeroed:
        scalar.backward()
        with pytest.raises(AssertionError):
            assert_zero_grad(module)
    else:
        assert_zero_grad(module)
