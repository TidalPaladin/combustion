#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import torch

from combustion.util import check_is_tensor


@pytest.mark.parametrize(
    "inputs,is_tensor",
    [
        pytest.param("foo", False, id="foo"),
        pytest.param(torch.rand(10), True, id="torch.rand()"),
        pytest.param(None, False, id="None"),
    ],
)
def test_is_tensor(inputs, is_tensor):
    if not is_tensor:
        with pytest.raises(TypeError):
            check_is_tensor(inputs, "inputs")
    else:
        check_is_tensor(inputs, "inputs")
