#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import torch
from torch import Tensor

from combustion.util import alpha_blend, apply_colormap


class TestApplyColormap:
    @pytest.mark.parametrize("dtype", ["float", "long", "byte"])
    @pytest.mark.parametrize(
        "inputs",
        [
            pytest.param(torch.rand(1, 1, 10, 10), id="2D"),
            pytest.param(torch.rand(1, 1, 10, 10, 10), id="3D"),
            pytest.param(torch.rand(1, 1, 10), id="1D"),
        ],
    )
    def test_input_output_shape(self, inputs, dtype):
        if dtype == "long":
            inputs = inputs.mul(1024).long()
        elif dtype == "byte":
            inputs = inputs.mul(255).byte()

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

    @pytest.mark.usefixtures("cuda_or_skip")
    def test_cuda_tensor(self):
        inputs = torch.rand(1, 1, 10, 10).cuda()
        output = apply_colormap(inputs, "gray")
        assert isinstance(output, Tensor)
        assert output.device != "cpu"


class TestAlphaBlend:
    @pytest.mark.parametrize(
        "shape",
        [
            pytest.param((1, 1, 10, 10), id="2D"),
            pytest.param((1, 1, 10, 10, 10), id="3D"),
            pytest.param((1, 1, 10), id="1D"),
            pytest.param((1, 3, 10, 10), id="RGB"),
        ],
    )
    def test_input_shapes(self, shape):
        dest = torch.rand(*shape)
        src = torch.rand(*shape)
        out, _ = alpha_blend(src, dest)
        assert out.shape == dest.shape

    def test_output_alpha(self):
        dest = torch.rand(1, 1, 10, 10)
        src = torch.rand(1, 1, 10, 10)
        _, out_alpha = alpha_blend(src, dest)
        assert (out_alpha == 1).all()

    def test_output_channels(self, cuda):
        dest = torch.zeros(1, 1, 10, 10).float()
        src = torch.zeros_like(dest).float()

        if cuda:
            dest = dest.cuda()
            src = src.cuda()

        dest[0, 0, 0, 0] = 1.0
        src[0, 0, 1, 1] = 1.0

        out, _ = alpha_blend(src, dest)
        assert out.device == dest.device
        assert out[0, 0, 0, 0] == 0.5
        assert out[0, 0, 1, 1] == 0.5

    def test_non_float_input(self):
        dest = torch.zeros(1, 1, 10, 10).byte()
        src = torch.zeros_like(dest)

        dest[0, 0, 0, 0] = 255
        src[0, 0, 1, 1] = 255

        out, _ = alpha_blend(src, dest)
        assert out.dtype == torch.uint8
        assert out[0, 0, 0, 0] == 127
        assert out[0, 0, 1, 1] == 127
