#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch

from combustion.nn.functional import complete_iou_loss


class TestCompleteIouLossFunctional:
    def test_basic_example(self):
        inputs = torch.tensor(
            [
                [1, 1, 10, 10],
                [2, 2, 4, 4],
                [2, 1, 4, 3],
                [1, 1, 3, 3],
            ]
        )

        targets = torch.tensor(
            [
                [1, 1, 10, 10],
                [0, 0, 4, 4],
                [0, 0, 4, 4],
                [0, 0, 4, 4],
            ]
        )

        loss = complete_iou_loss(inputs, targets, reduction="none")
        expected = torch.tensor([0.00, 0.8125, 0.7812, 0.7500])
        assert torch.allclose(loss, expected, atol=0.001)

    def test_differentiable(self):
        inputs = torch.tensor(
            [
                [1, 1, 10, 10],
                [2, 2, 4, 4],
                [2, 1, 4, 3],
                [1, 1, 3, 3],
            ],
            dtype=torch.float,
        )
        targets = torch.tensor(
            [
                [1, 1, 10, 10],
                [0, 0, 4, 4],
                [0, 0, 4, 4],
                [0, 0, 4, 4],
            ],
            dtype=torch.float,
        )

        inputs.requires_grad = True
        loss = complete_iou_loss(inputs, targets)
        loss.sum().backward()
