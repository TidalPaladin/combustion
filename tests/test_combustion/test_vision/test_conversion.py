#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import torch

from combustion.vision import to_8bit


@pytest.mark.parametrize("per_channel", [True, False])
def test_to_8bit(per_channel):
    img = torch.rand(3, 32, 32)
    result = to_8bit(img, per_channel)
    assert result.shape == img.shape
    assert result.min() >= 0
    assert result.max() <= 255
    assert result.dtype == torch.uint8
