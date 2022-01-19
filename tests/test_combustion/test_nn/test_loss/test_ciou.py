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

    def test_zero_loss(self, criterion):
        torch.random.manual_seed(42)
        inputs = torch.rand(32, 4) * 100
        targets = inputs.clone()
        loss = criterion(inputs, targets)
        assert torch.allclose(loss, torch.zeros_like(loss))

    def test_random_example(self, criterion):
        torch.random.manual_seed(42)
        inputs = torch.rand(32, 4)
        targets = torch.rand(32, 4)
        loss = criterion(inputs, targets)
        assert not loss.isnan().any()

    def test_is_differentiable(self, cuda, criterion):
        torch.random.manual_seed(42)
        inputs = torch.rand(32, 4, requires_grad=True)
        targets = torch.rand(32, 4)

        if cuda:
            inputs = inputs.cuda.half()
            targets = targets.cuda()

        loss = criterion(inputs, targets)
        loss.sum().backward()

    def test_increasing_loss_center_distance(self):
        torch.random.manual_seed(42)
        targets = torch.rand(32, 4) * 100
        input1 = targets.clone()

        input2 = targets.clone()
        input2[..., :2].add_(10)

        input3 = targets.clone()
        input3[..., :2].add_(20)

        loss1 = complete_iou_loss(input1, targets)
        loss2 = complete_iou_loss(input2, targets)
        loss3 = complete_iou_loss(input3, targets)

        assert (loss2 > loss1).all()
        assert (loss3 > loss2).all()

    def test_increasing_loss_area_mismatch(self):
        torch.random.manual_seed(42)
        targets = torch.rand(32, 4) * 100
        input1 = targets.clone()

        input2 = targets.clone()
        input2[..., 2:].add_(10)

        input3 = targets.clone()
        input3[..., 2:].add_(20)

        input4 = targets.clone()
        input4[..., 2:].sub_(10).clamp_min_(1e-4)

        input5 = targets.clone()
        input5[..., 2:].sub_(20).clamp_min_(1e-4)


        loss1 = complete_iou_loss(input1, targets)
        loss2 = complete_iou_loss(input2, targets)
        loss3 = complete_iou_loss(input3, targets)
        loss4 = complete_iou_loss(input4, targets)
        loss5 = complete_iou_loss(input5, targets)

        assert (loss2 > loss1).all()
        assert (loss3 > loss2).all()
        assert (loss4 > loss1).all()
        assert (loss5 > loss4).all()

    def test_loss_no_intersection(self):
        targets = torch.tensor([
            [10, 10, 2, 2],
            [20, 20, 2, 2],
        ])
        inputs1 = torch.tensor([
            [20, 20, 2, 2],
            [10, 10, 2, 2],
        ])
        inputs2 = torch.tensor([
            [15, 15, 2, 2],
            [15, 15, 2, 2],
        ])

        loss1 = complete_iou_loss(inputs1, targets)
        loss2 = complete_iou_loss(inputs2, targets)
        assert (loss2 < loss1).all()

    def test_scale_invariant(self, criterion):
        torch.random.manual_seed(42)
        inputs = torch.rand(32, 4)
        targets = torch.rand(32, 4)
        loss1 = criterion(inputs, targets)
        loss2 = criterion(inputs*1000, targets*1000)
        assert torch.allclose(loss1, loss2)


    def test_large_boxes(self):
        torch.random.manual_seed(42)
        targets = torch.rand(8, 4)
        targets[..., -2:].div_(10)

        base_input = torch.rand_like(targets)
        base_input[..., :2] = targets[..., :2]

        input1 = base_input.clone()
        input1[..., -2:].div_(10)

        input2 = base_input.clone()

        input3 = base_input.clone()
        input3[..., -2:].clamp_min_(0.9)

        loss1 = complete_iou_loss(input1, targets)
        loss2 = complete_iou_loss(input2, targets)
        loss3 = complete_iou_loss(input3, targets)

        assert (loss2 > loss1).all()
        assert (loss3 > loss2).all()
