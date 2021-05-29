#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
from torch import Tensor
import torch

from combustion.nn.modules.transformer.performer import generalized_kernel_features

class TestKernels:

    def test_generalized_kernel(self):
        L, N, E = 10, 3, 20
        R = 5
        data = torch.rand(L, N, E)
        projection = torch.rand(E, R)
        kernel_func = lambda x: x

        features = generalized_kernel_features(data, kernel_func, projection)
        assert isinstance(features, Tensor)
