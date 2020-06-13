#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import torch

from combustion.vision import to_8bit


class TestTo8bit:
    @pytest.fixture(params=["float", "long", "int"])
    def dtype(self, request):
        return request.param

    @pytest.fixture(
        params=[(3, 32, 32), (1, 32, 32), (2, 1, 32, 32),]
    )
    def shape(self, request):
        return request.param

    @pytest.fixture
    def input(self, shape, dtype):
        torch.random.manual_seed(42)
        img = torch.rand(*shape)

        if dtype == "float":
            return img

        img = img.mul_(10000)
        if dtype == "long":
            return img.long()
        elif dtype == "int":
            return img.int()
        elif dtype == "byte":
            return img.byte()
        else:
            raise pytest.UsageError(f"Unknown dtype {dtype}")

    @pytest.mark.parametrize("per_channel", [True, False])
    def test_to_8bit(self, input, per_channel, shape, dtype):
        result = to_8bit(input, per_channel)
        assert result.shape == shape
        assert result.min() >= 0
        assert result.max() <= 255
        assert result.unique().numel() >= 10
        assert result.dtype == torch.uint8

        if dtype == "byte":
            assert torch.allclose(input, result)
