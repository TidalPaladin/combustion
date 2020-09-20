#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import torch

from combustion.vision.filters import CLAHE


@pytest.mark.parametrize("num_channels", [1, 3])
def test_clahe(num_channels):
    torch.random.manual_seed(42)
    inputs = torch.rand(1, num_channels, 32, 64).mul_(255).byte()
    xform = CLAHE(clipLimit=2.0, tileGridSize=(5, 5))
    output = xform(inputs)

    assert output.shape == inputs.shape
    assert not torch.allclose(inputs, output)
