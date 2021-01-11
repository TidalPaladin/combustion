#!/usr/bin/env python
# -*- coding: utf-8 -*-

import timeit

import pytest
import torch
from packaging import version

from combustion.vision.filters import RelativeIntensity, relative_intensity


has_torch18 = version.parse(torch.__version__) > version.parse("1.7.1")


@pytest.fixture
def inputs():
    inputs = torch.zeros(1, 1, 5, 5)
    inputs[..., 2, 2] = 1
    return inputs


@pytest.mark.skipif(not has_torch18, reason="torch>=1.8 is required")
class TestRelativeIntensityFunctional:
    def test_intensity_single_kernel(self, inputs):
        kernel = [
            (3, 3),
        ]
        sigma = [
            (1, 1),
        ]
        output = relative_intensity(inputs, kernel, sigma)
        expected = torch.tensor(
            [
                [
                    [
                        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                        [0.0000, -0.0751, -0.1238, -0.0751, 0.0000],
                        [0.0000, -0.1238, 0.7958, -0.1238, 0.0000],
                        [0.0000, -0.0751, -0.1238, -0.0751, 0.0000],
                        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                    ]
                ]
            ]
        )
        assert torch.allclose(output, expected, atol=1e-4)

    @pytest.mark.parametrize("combine", ["min", "max", "sum", "mean"])
    def test_intensity_combined_kernels(self, inputs, combine):
        kernel = [(3, 3), (5, 5)]
        sigma = [(1, 1), (2, 2)]
        actual = relative_intensity(inputs, kernel, sigma, combine=combine)
        v1 = relative_intensity(inputs, [kernel[0]], [sigma[0]])
        v2 = relative_intensity(inputs, [kernel[1]], [sigma[1]])

        if combine == "min":
            expected = torch.min(v1, v2)
        elif combine == "max":
            expected = torch.max(v1, v2)
        elif combine == "mean":
            expected = torch.add(v1, v2).div_(2)
        elif combine == "sum":
            expected = torch.add(v1, v2)

        assert torch.allclose(actual, expected, atol=1e-4)

    @pytest.mark.ci_skip
    def test_time(self):
        kernel = [(101, 101)]
        sigma = [(50, 50)]
        inputs = torch.rand(1, 1, 2048, 1024)

        def func():
            relative_intensity(inputs, kernel, sigma)

        t = timeit.timeit(func, number=2)
        print(f"Relative intensity runtime: {t}")
        assert t <= 0.1


@pytest.mark.skipif(not has_torch18, reason="torch>=1.8 is required")
class TestRelativeIntensityClass:
    def test_intensity_single_kernel(self, inputs):
        kernel = [
            (3, 3),
        ]
        sigma = [
            (1, 1),
        ]
        layer = RelativeIntensity(kernel, sigma)
        output = layer(inputs)
        expected = torch.tensor(
            [
                [
                    [
                        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                        [0.0000, -0.0751, -0.1238, -0.0751, 0.0000],
                        [0.0000, -0.1238, 0.7958, -0.1238, 0.0000],
                        [0.0000, -0.0751, -0.1238, -0.0751, 0.0000],
                        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                    ]
                ]
            ]
        )
        assert torch.allclose(output, expected, atol=1e-4)

    @pytest.mark.parametrize("combine", ["min", "max", "sum", "mean"])
    def test_intensity_combined_kernels(self, inputs, combine):
        kernel = [(3, 3), (5, 5)]
        sigma = [(1, 1), (2, 2)]
        layer = RelativeIntensity(kernel, sigma, combine=combine)
        actual = layer(inputs)
        v1 = relative_intensity(inputs, [kernel[0]], [sigma[0]])
        v2 = relative_intensity(inputs, [kernel[1]], [sigma[1]])

        if combine == "min":
            expected = torch.min(v1, v2)
        elif combine == "max":
            expected = torch.max(v1, v2)
        elif combine == "mean":
            expected = torch.add(v1, v2).div_(2)
        elif combine == "sum":
            expected = torch.add(v1, v2)

        assert torch.allclose(actual, expected, atol=1e-4)
