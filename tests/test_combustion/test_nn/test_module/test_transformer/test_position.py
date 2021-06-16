#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
from torch import Tensor
import pytest
from combustion.nn.modules.transformer.position import RelativePositionalEncoder

class TestRelativePositionalEncoder:

    def test_forward(self):
        B, N = 2, 100
        coords = torch.normal(0, 10.0, (B, N, 3))

        l = RelativePositionalEncoder(3, 32)
        assert False
