#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch

from combustion.nn import Standardize


class TestStandardize:
    def test_basic_input(self):
        input = torch.tensor(
            [
                [[0.0, 1.0], [1.0, 2.0]],
                [[0.0, 1.0], [1.0, 2.0]],
                [[0.0, 1.0], [1.0, 2.0]],
            ]
        )

        expected = torch.tensor(
            [
                [[-1.5, 0.0], [0.0, 1.5]],
                [[-1.5, 0.0], [0.0, 1.5]],
                [[-1.5, 0.0], [0.0, 1.5]],
            ]
        )
        layer = Standardize(dims=(-1, -2))
        output = layer(input)
        assert torch.allclose(output, expected)

    def test_non_contiguous_dims(self):
        input = torch.tensor(
            [
                [[0.0, 1.0], [1.0, 2.0]],
                [[0.0, 1.0], [1.0, 2.0]],
                [[0.0, 1.0], [1.0, 2.0]],
            ]
        ).permute(1, 0, 2)

        expected = torch.tensor(
            [
                [[-1.5, 0.0], [0.0, 1.5]],
                [[-1.5, 0.0], [0.0, 1.5]],
                [[-1.5, 0.0], [0.0, 1.5]],
            ]
        ).permute(1, 0, 2)
        layer = Standardize(dims=(0, 2))
        output = layer(input)
        assert torch.allclose(output, expected)

    def test_zero_variance(self):
        input = torch.tensor(
            [
                [[0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0]],
            ]
        )

        layer = Standardize(dims=(-1, -2))
        output = layer(input)
        assert not torch.isnan(output).any(), "divided by zero because variance=0"

    def test_repr(self):
        layer = Standardize(dims=(-1, -2))
        print(layer)
