#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import torch
from torch import Tensor

from combustion.nn import OCR
from combustion.testing import TorchScriptTestMixin


class TestOCR(TorchScriptTestMixin):
    @pytest.fixture(params=[5, 10])
    def in_channels(self, request):
        return request.param

    @pytest.fixture
    def num_classes(self):
        return 5

    @pytest.fixture(params=[1, 2])
    def downsample(self, request):
        return request.param

    @pytest.fixture(params=[3, 5, 10])
    def key_channels(self, request):
        return request.param

    @pytest.fixture
    def pixels(self, in_channels):
        torch.random.manual_seed(42)
        return torch.rand(1, in_channels, 32, 32)

    @pytest.fixture
    def regions(self, in_channels):
        torch.random.manual_seed(42)
        return torch.rand(1, in_channels, 32, 32)

    @pytest.fixture
    def model(self):
        return OCR(5, 10, 3)

    def test_construct(self, in_channels, num_classes, key_channels):
        OCR(in_channels, key_channels, num_classes)

    def test_forward(self, num_classes, in_channels, key_channels, pixels, regions):
        l = OCR(in_channels, key_channels, num_classes)
        output = l(pixels, regions)
        assert isinstance(output, Tensor)
        assert output.shape[0] == pixels.shape[0]
        assert output.shape[1] == num_classes
        assert output.shape[2:] == pixels.shape[2:]

    def test_backward(self, num_classes, in_channels, key_channels, pixels, regions):
        pixels.requires_grad = True
        regions.requires_grad = True
        l = OCR(in_channels, key_channels, num_classes)
        output = l(pixels, regions)
        scalar = output.sum()
        scalar.backward()

    def test_create_region_target(self, num_classes):
        target = torch.randint(0, num_classes, (3, 32, 32))
        output = OCR.create_region_target(target, num_classes)

        assert isinstance(output, Tensor)
        assert tuple(output.shape) == (3, num_classes, 32, 32)
        assert output.unique().numel() == 2
        assert output.max() == 1
        assert output.min() == 0
