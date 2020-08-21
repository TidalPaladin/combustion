#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import torch
from torch import Tensor

from combustion.util import apply_colormap


class TestApplyColormap:
    @pytest.mark.parametrize(
        "inputs",
        [
            pytest.param(torch.rand(1, 1, 10, 10), id="2D"),
            pytest.param(torch.rand(1, 1, 10, 10, 10), id="3D"),
            pytest.param(torch.rand(1, 1, 10), id="1D"),
        ],
    )
    def test_input_output_shape(self, inputs):
        out = apply_colormap(inputs)
        assert isinstance(out, Tensor)
        assert out.shape[2:] == inputs.shape[2:]
        assert out.shape[0] == inputs.shape[0]
        assert out.shape[1] == 4

    def test_cmap_locations(self):
        inputs = torch.rand(1, 1, 10, 10)
        inputs2 = torch.rand(1, 1, 10, 10)
        out1 = apply_colormap(inputs, "gray")
        out2 = apply_colormap(inputs2, "gray")

        greater_input = inputs <= inputs2
        greater_output = out1[:, 0, ...] <= out2[:, 0, ...]
        assert ~(torch.logical_xor(greater_input, greater_output)).all()
