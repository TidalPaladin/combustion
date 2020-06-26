#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import torch

from combustion.nn import CenterNetLoss


class TestConstantInputs:
    @pytest.fixture(params=[True, False])
    def box_present(self, request):
        return request.param

    @pytest.fixture
    def input(self):
        return torch.tensor(
            [
                [[1.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 1.0]],
                [[1.0, 1.0], [1.0, 1.0]],
                [[2.0, 2.0], [2.0, 2.0]],
                [[3.0, 3.0], [3.0, 3.0]],
                [[4.0, 4.0], [4.0, 4.0]],
            ]
        )

    @pytest.fixture
    def target(self, box_present):
        if box_present:
            _ = torch.tensor(
                [
                    [[0.0, 0.0], [0.0, 0.0]],
                    [[1.0, 0.0], [0.0, 1.0]],
                    [[-2.0, -1.0], [-1.0, 2.0]],
                    [[-3.0, -1.0], [-1.0, 3.0]],
                    [[4.0, -1.0], [-1.0, 4.0]],
                    [[5.0, -1.0], [-1.0, 5.0]],
                ]
            )
        else:
            _ = torch.tensor(
                [
                    [[0.0, 0.0], [0.0, 0.0]],
                    [[0.0, 0.0], [0.0, 0.0]],
                    [[-1.0, -1.0], [-1.0, -1.0]],
                    [[-1.0, -1.0], [-1.0, -1.0]],
                    [[-1.0, -1.0], [-1.0, -1.0]],
                    [[-1.0, -1.0], [-1.0, -1.0]],
                ]
            )
        return _

    @pytest.fixture
    def expected(self, target, box_present):

        cls_loss = torch.tensor(
            [[[0.7018683, 0.1732868], [0.1732868, 0.1732868]], [[0.1732868, 0.1732868], [0.1732868, 0.0226581]]]
        )

        reg_loss = torch.tensor(
            [
                [[2.5000, 0.0000], [0.0000, 0.5000]],
                [[4.5000, 0.0000], [0.0000, 0.5000]],
                [[0.5000, 0.0000], [0.0000, 0.5000]],
                [[0.5000, 0.0000], [0.0000, 0.5000]],
            ]
        )

        if not box_present:
            reg_loss[:, 0, 0] = 0
            reg_loss[:, 1, 1] = 0
            cls_loss[1, 1, 1] = 0.7018683
        return cls_loss, reg_loss

    def test_returned_correct_class_loss(self, input, target, expected):
        criterion = CenterNetLoss(reduction="none")
        loss = criterion(input, target)
        assert len(loss) == 2
        cls_loss, reg_loss = loss
        assert torch.allclose(cls_loss, expected[0])

    def test_returned_correct_regression_loss(self, input, target, expected):
        criterion = CenterNetLoss(reduction="none")
        loss = criterion(input, target)
        assert len(loss) == 2
        cls_loss, reg_loss = loss
        assert torch.allclose(reg_loss, expected[1])

    @pytest.mark.parametrize("reduction", ["mean", "sum"])
    def test_reduction(self, input, target, box_present, reduction):
        baseline = CenterNetLoss(reduction="none")
        criterion = CenterNetLoss(reduction=reduction)

        num_boxes = 2 if box_present else 0

        true_loss = baseline(input, target)
        loss = criterion(input, target)

        if reduction == "sum":
            true_loss = [x.sum() for x in true_loss]
        else:
            true_loss = [x.sum() / max(num_boxes, 1) for x in true_loss]

        assert torch.allclose(loss[0], true_loss[0])
        assert torch.allclose(loss[1], true_loss[1])


class TestRandomInputs:
    @pytest.fixture(params=[None, 1, 2])
    def batch_size(self, request):
        return request.param

    @pytest.fixture(params=[(32, 32), (64, 32)])
    def input_shape(self, request):
        return request.param

    @pytest.fixture(params=[2, 4])
    def num_classes(self, request):
        return request.param

    @pytest.fixture
    def input(self, batch_size, input_shape, num_classes):
        torch.random.manual_seed(42)
        if batch_size is not None:
            cls = torch.rand(batch_size, num_classes, *input_shape)
            reg = torch.randint(0, 10, (batch_size, 4, *input_shape))
        else:
            cls = torch.rand(num_classes, *input_shape)
            reg = torch.randint(0, 10, (4, *input_shape))
        return torch.cat([cls.float(), reg.float()], dim=-3)

    @pytest.fixture
    def target(self, batch_size, input_shape, num_classes):
        torch.random.manual_seed(21)
        if batch_size is not None:
            cls = torch.rand(batch_size, num_classes, *input_shape)
            reg = torch.randint(0, 10, (batch_size, 4, *input_shape))
        else:
            cls = torch.rand(num_classes, *input_shape)
            reg = torch.randint(0, 10, (4, *input_shape))

        maxes = cls.max(dim=-1, keepdim=True).values.max(dim=-2, keepdim=True).values
        cls.div_(maxes)
        return torch.cat([cls.float(), reg.float()], dim=-3)

    def test_center_net_loss(self, input, target):
        criterion = CenterNetLoss(reduction="none")
        loss = criterion(input, target)
        assert len(loss) == 2
        cls_loss, reg_loss = loss
        assert cls_loss.bool().any()

    def test_smooth_l1(self, input, target):
        baseline = CenterNetLoss(reduction="none")
        criterion = CenterNetLoss(reduction="none", smooth=False)

        true_loss = baseline(input, target)
        loss = criterion(input, target)

        assert torch.allclose(loss[0], true_loss[0])
        assert not torch.allclose(loss[1], true_loss[1])
