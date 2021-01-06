#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import torch

from combustion.nn import CompleteIoULoss
from combustion.nn.functional import complete_iou_loss


class TestCompleteIouLoss:
    @pytest.fixture(
        params=[
            pytest.param(True, id="functional"),
            pytest.param(False, id="class"),
        ]
    )
    def criterion(self, request):
        if request.param:
            return lambda x, y: complete_iou_loss(x, y, reduction="none")
        else:
            return CompleteIoULoss(reduction="none")

    def test_basic_example(self, criterion):
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

        loss = criterion(inputs, targets)
        expected = torch.tensor([0.00, 0.8125, 0.7812, 0.7500])
        assert torch.allclose(loss, expected, atol=0.001)

    def test_random_example(self, criterion):
        inputs = torch.rand(32, 4)
        targets = torch.rand(32, 4)
        loss = criterion(inputs, targets)
        assert not loss.isnan().any()

    def test_bad_example(self, criterion):
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

        loss = criterion(inputs, targets)
        assert not loss.isnan().any()

    def test_differentiable(self, criterion):
        inputs = torch.tensor(
            [
                [1, 1, 10, 10],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
            ]
        ).float()

        targets = torch.tensor(
            [
                [1, 1, 10, 10],
                [3, 3, 1, 1],
                [3, 2, 1, 2],
                [2, 2, 2, 2],
            ]
        )
        inputs.requires_grad = True
        loss = criterion(inputs, targets)
        loss.sum().backward()

    def test_absolute(self):
        inputs1 = torch.tensor(
            [
                [1, 1, 10, 10],
                [1, 1, 1, 1],
            ]
        )
        targets1 = torch.tensor(
            [
                [1, 1, 10, 10],
                [3, 3, 1, 1],
            ]
        )

        inputs2 = torch.tensor(
            [
                [0, 0, 11, 11],
                [2, 2, 4, 4],
            ]
        )
        targets2 = torch.tensor(
            [
                [0, 0, 11, 11],
                [0, 0, 4, 4],
            ]
        )

        loss1 = complete_iou_loss(inputs1, targets1)
        loss2 = complete_iou_loss(inputs2, targets2, absolute=True)
        assert torch.allclose(loss1, loss2, atol=0.001)
