#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import torch
from torch.testing import assert_allclose


def assert_has_gradient(module, recurse=True):
    __tracebackhide__ = True

    if isinstance(module, torch.Tensor) and module.grad is None:
        pytest.fail()
    elif isinstance(module, torch.nn.Module):
        for name, param in module.named_parameters(recurse=recurse):
            if param.requires_grad and param.grad is None:
                pytest.fail()


def assert_zero_grad(module, recurse=True):
    __tracebackhide__ = True

    if isinstance(module, torch.Tensor) and not all(module.grad == 0):
        pytest.fail()
    elif isinstance(module, torch.nn.Module):
        for name, param in module.named_parameters(recurse=recurse):
            if param.requires_grad and not all(param.grad == 0):
                pytest.fail()


def assert_in_training_mode(module):
    __tracebackhide__ = True
    if not module.training:
        pytest.fail("Module not in training mode")


def assert_in_eval_mode(module):
    __tracebackhide__ = True
    if module.training:
        pytest.fail("Module not in eval mode")


def assert_tensors_close(x, y, *args, **kwargs):
    __tracebackhide__ = True
    msg = None
    try:
        assert_allclose(x, y, *args, **kwargs)
        return
    except AssertionError as e:
        msg = str(e)
    pytest.fail(msg)


def assert_is_int_tensor(x):
    __tracebackhide__ = True
    if not torch.allclose(x, x.round()):
        try:
            assert str(x) == str(x.round())
        except AssertionError as e:
            pytest.fail(str(e))


__all__ = [
    "assert_has_gradient",
    "assert_zero_grad",
    "assert_is_int_tensor",
    "assert_in_training_mode",
    "assert_in_eval_mode",
    "assert_tensors_close",
]
