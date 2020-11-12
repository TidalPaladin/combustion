#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch

from combustion.nn.functional import complete_iou_loss


class TestCompleteIouLossFunctional:
    def test_basic_example(self):
        inputs = torch.tensor(
            [
                [1, 1, 10, 10],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
            ]
        )

        targets = torch.tensor(
            [
                [1, 1, 10, 10],
                [3, 3, 1, 1],
                [3, 2, 1, 2],
                [2, 2, 2, 2],
            ]
        )

        loss = complete_iou_loss(inputs, targets, reduction="none")
        expected = torch.tensor([0.00, 0.8125, 0.7812, 0.7500])
        assert torch.allclose(loss, expected, atol=0.001)

    def test_random_example(self):
        inputs = torch.rand(32, 4)
        targets = torch.rand(32, 4)
        loss = complete_iou_loss(inputs, targets, reduction="none")
        assert not loss.isnan().any()

    def test_bad_example(self):
        inputs = torch.tensor(
            [
                [1, 1, 1, 1],
                [0, 0, 0, 0],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
            ]
        )

        targets = torch.tensor(
            [
                [-1, -1, -1, -1],
                [-1, -1, -1, -1],
                [-1, -1, -1, -1],
                [-1, -1, -1, -1],
                [1, 1, 1, 1],
            ]
        )

        loss = complete_iou_loss(inputs, targets, reduction="none")
        assert not loss.isnan().any()

    def test_differentiable(self):
        inputs = torch.tensor(
            [
                [1, 1, 10, 10],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
            ]
        )

        targets = torch.tensor(
            [
                [1, 1, 10, 10],
                [3, 3, 1, 1],
                [3, 2, 1, 2],
                [2, 2, 2, 2],
            ]
        )
        inputs.requires_grad = True
        loss = complete_iou_loss(inputs, targets)
        loss.sum().backward()
