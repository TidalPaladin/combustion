#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import torch

from combustion.vision import to_8bit


@pytest.mark.parametrize("shape", [(3, 32, 32), (1, 32, 32), (2, 1, 32, 32),])
@pytest.mark.parametrize("per_channel", [True, False])
def test_to_8bit(per_channel, shape):
    img = torch.rand(*shape)
    result = to_8bit(img, per_channel)
    assert result.shape == shape
    assert result.min() >= 0
    assert result.max() <= 255
    assert result.dtype == torch.uint8
