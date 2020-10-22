#!/usr/bin/env python
# -*- coding: utf-8 -*-

from numbers import Number

import pytest
import torch
from torch import Tensor

from combustion.util import percent_change, percent_error_change


@pytest.mark.parametrize("is_tensor", [True, False])
@pytest.mark.parametrize(
    "old,new",
    [
        pytest.param(0.8, 0.9),
        pytest.param(-0.8, -0.9),
        pytest.param(0.8, -0.9),
        pytest.param(0.8, -0.9),
        pytest.param(0.5, 1.0),
        pytest.param(100, 200),
    ],
)
def test_percent_change(old, new, is_tensor):
    if is_tensor:
        old = torch.tensor(old)
        new = torch.tensor(new)

    result = percent_change(old, new)
    expected = 100.0 * (new - old) / abs(old)

    if is_tensor:
        assert isinstance(result, Tensor)
    else:
        assert isinstance(result, Number)

    result = torch.as_tensor(result)
    expected = torch.as_tensor(expected)
    assert torch.allclose(result, expected, rtol=1e-5)


@pytest.mark.parametrize("is_tensor", [True, False])
@pytest.mark.parametrize(
    "old,new,max_val",
    [
        pytest.param(0.8, 0.9, 1.0),
        pytest.param(-0.8, -0.9, 1.0),
        pytest.param(0.8, -0.9, 1.0),
        pytest.param(0.8, -0.9, 1.0),
        pytest.param(0.5, 1.0, 1.0),
        pytest.param(100, 200, 300),
    ],
)
def test_percent_error_change(old, new, is_tensor, max_val):
    if is_tensor:
        old = torch.tensor(old)
        new = torch.tensor(new)

    result = percent_error_change(old, new, max_val=max_val)
    err_old = max_val - old
    err_new = max_val - new
    expected = 100.0 * (err_new - err_old) / abs(err_old)

    if is_tensor:
        assert isinstance(result, Tensor)
    else:
        assert isinstance(result, Number)

    result = torch.as_tensor(result)
    expected = torch.as_tensor(expected)
    assert torch.allclose(result, expected, rtol=1e-5)
