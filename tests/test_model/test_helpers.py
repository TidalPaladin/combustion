#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest


regional_max = pytest.importorskip("combustion.model.head.regional_max")


class TestRegionalMax:
    @pytest.mark.parametrize(
        "shape",
        [
            pytest.param((1, 1, 12, 12), id="1x1x12x12"),
            pytest.param((8, 1, 12, 12), id="8x1x12x12"),
            pytest.param((3, 3, 12, 12), id="3x3x12x12"),
            pytest.param((1, 1, 11, 11), id="1x1x11x11"),
        ],
    )
    def test_shape_unchanged(self, torch, shape):
        input = torch.rand(*shape)
        output = regional_max(input, 3)
        assert input.shape == output.shape

    def test_regional_max_unchanged(self, torch):
        input = (
            torch.Tensor(
                [
                    [0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                ]
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )
        output = regional_max(input, 3)
        assert input.shape == output.shape
        assert output.equal(input)

    def test_output(self, torch):
        input = (
            torch.Tensor(
                [
                    [0, 0.1, 0.05, 0, 0, 0],
                    [0, 0.1, 0.25, 0, 1, 0],
                    [0, 0, 0.05, 0, 0, 0],
                    [0, 0, 0.2, 0, 0.05, 0],
                    [0, 0, 0.05, 0, 0, 0],
                    [0.3, 0, 0.2, 0, 0, 0.1],
                ]
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )
        expected = (
            torch.Tensor(
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0.3, 0, 0, 0, 0, 0.1],
                ]
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )
        output = regional_max(input, 3)
        assert expected.shape == output.shape
        assert output.equal(expected)
        print(output)

    def test_output2(self, torch):
        input = (
            torch.Tensor(
                [
                    [1.0, 0.1, 0.05, 0, 0, 0],
                    [0, 0.1, 0.25, 0.26, 1, 0],
                    [2.0, 0, 0.05, 0, 0, 0],
                    [0, 0, 0.2, 0, 0.05, 0],
                    [0, 0, 0.05, 0, 0, 0],
                    [0.3, 0, 0.2, 0, 0, 0.1],
                ]
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )
        expected = (
            torch.Tensor(
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0],
                    [2.0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0.3, 0, 0, 0, 0, 0.1],
                ]
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )
        output = regional_max(input, 3)
        assert expected.shape == output.shape
        assert output.equal(expected)
        print(output)

    def test_output3(self, torch):
        input = (
            torch.Tensor(
                [
                    [1.0, 0.1, 0.05, 0, 0, 0],
                    [0, 0.1, 0.25, 0.26, 1, 0],
                    [2.0, 0, 0.05, 0, 0, 0],
                    [2.0, 0, 0.2, 0, 0.05, 0],
                    [0, 0, 0.05, 0, 0, 0],
                    [0.3, 0, 0.2, 0, 0, 0.1],
                ]
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )
        expected = (
            torch.Tensor(
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0],
                    [2.0, 0, 0, 0, 0, 0],
                    [2.0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                ]
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )
        output = regional_max(input, 4)
        assert expected.shape == output.shape
        assert output.equal(expected)
        print(output)

    def test_output4(self, torch):
        input = (
            torch.Tensor(
                [[0, 0, 0, 0, 0], [0.4, 0.55, 0.2, 0, 0], [0.1, 0.6, 0.4, 0, 0], [0, 0.4, 0, 0, 0], [0, 0, 0, 0, 0],]
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )
        expected = (
            torch.Tensor([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0.6, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0],])
            .unsqueeze(0)
            .unsqueeze(0)
        )
        output = regional_max(input, 4)
        assert expected.shape == output.shape
        assert output.equal(expected)
        print(output)
