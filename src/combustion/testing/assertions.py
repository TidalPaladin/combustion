#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch import Tensor
from torch.testing import assert_allclose


def assert_has_gradient(module: nn.Module, recurse: bool = True):
    r"""Asserts that the parameters in a module have ``requires_grad=True`` and
    that the gradient exists.

    Args:

        module (torch.nn.Module):
            The module to inspect

        recurse (bool, optional):
            Whether or not to recursively run the same assertion on the gradients
            of child modules.

    """
    __tracebackhide__ = True

    if isinstance(module, torch.Tensor) and module.grad is None:
        raise AssertionError(f"tensor grad == {module.grad}")
    elif isinstance(module, torch.nn.Module):
        for name, param in module.named_parameters(recurse=recurse):
            if param.requires_grad and param.grad is None:
                raise AssertionError(f"param {name} grad == {param.grad}")


def assert_zero_grad(module: nn.Module, recurse: bool = True):
    r"""Asserts that the parameters in a module have zero gradients.
    Useful for checking if `Optimizer.zero_grads()` was called.

    Args:

        module (torch.nn.Module):
            The module to inspect

        recurse (bool, optional):
            Whether or not to recursively run the same assertion on the gradients
            of child modules.

    """
    __tracebackhide__ = True

    if isinstance(module, torch.Tensor) and not all(module.grad == 0):
        raise AssertionError(f"module.grad == {module.grad}")
    elif isinstance(module, torch.nn.Module):
        for name, param in module.named_parameters(recurse=recurse):
            if param.requires_grad and not (param.grad is None or (~param.grad.bool()).all()):
                raise AssertionError(f"param {name} grad == {param.grad}")


def assert_in_training_mode(module: nn.Module):
    r"""Asserts that the module is in training mode, i.e. ``module.train()``
    was called

    Args:

        module (torch.nn.Module):
            The module to inspect

    """
    __tracebackhide__ = True
    if not module.training:
        raise AssertionError(f"module.training == {module.training}")


def assert_in_eval_mode(module: nn.Module):
    r"""Asserts that the module is in inference mode, i.e. ``module.eval()``
    was called.

    Args:

        module (torch.nn.Module):
            The module to inspect

    """
    __tracebackhide__ = True
    if module.training:
        raise AssertionError(f"module.training == {module.training}")


def assert_tensors_close(x: Tensor, y: Tensor, *args, **kwargs):
    r"""Asserts that the values two tensors are close. This is similar
    to :func:`torch.allclose`, but has cleaner output when used with
    pytest.

    Args:

        x (torch.Tensor):
            The first tensor.

        y (torch.Tensor):
            The second tensor.

    Additional positional or keyword args are passed to :func:`torch.allclose`.
    """
    __tracebackhide__ = True
    try:
        assert_allclose(x, y, *args, **kwargs)
        return
    except AssertionError as e:
        raise AssertionError(str(e))


def assert_is_int_tensor(x: Tensor):
    r"""Asserts that the values of a floating point tensor are integers.
    This test is equivalent to ``torch.allclose(x, x.round())``.

    Args:

        x (torch.Tensor):
            The first tensor.

        y (torch.Tensor):
            The second tensor.
    """
    __tracebackhide__ = True
    if not torch.allclose(x, x.round()):
        try:
            assert str(x) == str(x.round())
        except AssertionError as e:
            raise AssertionError(str(e))


__all__ = [
    "assert_has_gradient",
    "assert_zero_grad",
    "assert_is_int_tensor",
    "assert_in_training_mode",
    "assert_in_eval_mode",
    "assert_tensors_close",
]
