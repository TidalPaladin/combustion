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
    def test_random_input(self, input, per_channel, shape, dtype):
        result = to_8bit(input, per_channel)
        assert result.shape == shape
        assert result.min() >= 0
        assert result.max() <= 255
        assert result.unique().numel() >= 10
        assert result.dtype == torch.uint8

        if dtype == "byte":
            assert torch.allclose(input, result)

    def test_known_float_input_per_channel(self):
        input = torch.tensor([[-1.0, 0.0], [0.0, 1.0]]).unsqueeze(0)
        expected = torch.tensor([[0, 128], [128, 255]]).unsqueeze(0).byte()
        result = to_8bit(input, per_channel=True)
        assert result.shape == input.shape
        assert torch.allclose(result, expected)

    def test_known_long_input_per_channel(self):
        input = torch.tensor([[0, 1000], [1001, 2000]]).unsqueeze(0).long()
        expected = torch.tensor([[0, 128], [128, 255]]).unsqueeze(0).byte()
        result = to_8bit(input, per_channel=True)
        assert result.shape == input.shape
        assert torch.allclose(result, expected)

    @pytest.mark.parametrize("per_channel", [True, False])
    def test_known_input_per_channel_result(self, per_channel):
        input = torch.tensor([[[0, 1000], [1001, 2000]], [[0, 500], [500, 1000]],]).long()

        if per_channel:
            expected = torch.tensor([[[0, 128], [128, 255]], [[0, 128], [128, 255]]]).byte()
        else:
            expected = torch.tensor([[[0, 128], [128, 255]], [[0, 64], [64, 128]]]).byte()

        result = to_8bit(input, per_channel=per_channel)
        assert result.shape == input.shape
        assert torch.allclose(result, expected)

    def test_input_unchanged(self):
        input = torch.tensor([[0, 1000], [1001, 2000]]).unsqueeze(0).long()
        original_input = input.clone()
        result = to_8bit(input, per_channel=True)
        assert torch.allclose(input, original_input)
