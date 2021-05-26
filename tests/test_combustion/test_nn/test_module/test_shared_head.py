#!/usr/bin/env python
# -*- coding: utf-8 -*-


import pytest
import torch

from combustion.nn import SharedDecoder2d
from combustion.testing import TorchScriptTestMixin


class TestSharedDecoder(TorchScriptTestMixin):
    @pytest.fixture
    def model(self):
        return SharedDecoder2d(10, 5, 2, scaled=True)

    @pytest.fixture
    def data(self):
        base_channels = 10
        base_size = 64
        num_levels = 3

        fpn = [torch.rand(1, base_channels, base_size // (i + 1), base_size // (i + 1)) for i in range(num_levels)]
        return fpn

    @pytest.mark.parametrize(
        "in_channels,out_channels,num_convs,scaled,strides",
        [
            pytest.param(10, 5, 1, False, None),
            pytest.param(5, 10, 1, False, None),
            pytest.param(10, 5, 3, False, None),
            pytest.param(10, 5, 1, True, None),
            pytest.param(10, 5, 1, True, [8, 16, 32]),
        ],
    )
    def test_forward(self, in_channels, out_channels, num_convs, scaled, strides):
        model = SharedDecoder2d(in_channels, out_channels, num_convs, scaled, strides)

        base_size = 64
        num_levels = 3
        fpn = []
        for i in range(num_levels):
            size = base_size // (i + 1)
            fpn.append(torch.rand(1, in_channels, size, size, requires_grad=True))

        output = model(fpn)
        assert isinstance(output, list)
        assert len(output) == len(fpn)

        for level_idx, level in enumerate(output):
            assert level.shape[2:] == fpn[level_idx].shape[2:]
            assert level.shape[1] == out_channels

        scalar: Tensor = sum([x.sum() for x in output])  # type: ignore
        scalar.backward()
