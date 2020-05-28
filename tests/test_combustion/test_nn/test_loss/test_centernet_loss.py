#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import torch

from combustion.nn import CenterNetLoss


@pytest.fixture(params=[None, 1, 2])
def batch_size(request):
    return request.param


@pytest.fixture(params=[(32, 32), (64, 32)])
def input_shape(request):
    return request.param


@pytest.fixture(params=[2, 4])
def num_classes(request):
    return request.param


@pytest.fixture
def input(batch_size, input_shape, num_classes):
    torch.random.manual_seed(42)
    if batch_size is not None:
        cls = torch.rand(batch_size, num_classes, *input_shape)
        reg = torch.randint(0, 10, (batch_size, 4, *input_shape))
    else:
        cls = torch.rand(num_classes, *input_shape)
        reg = torch.randint(0, 10, (4, *input_shape))
    return torch.cat([cls.float(), reg.float()], dim=-3)


@pytest.fixture
def target(batch_size, input_shape, num_classes):
    torch.random.manual_seed(21)
    if batch_size is not None:
        cls = torch.rand(batch_size, num_classes, *input_shape)
        reg = torch.randint(0, 10, (batch_size, 4, *input_shape))
    else:
        cls = torch.rand(num_classes, *input_shape)
        reg = torch.randint(0, 10, (4, *input_shape))
    return torch.cat([cls.float(), reg.float()], dim=-3)


def test_center_net_loss(input, target):
    criterion = CenterNetLoss(reduction="none")
    loss = criterion(input, target)
    assert len(loss) == 2
    cls_loss, reg_loss = loss
    assert cls_loss.bool().any()
    assert reg_loss.bool().any()


def test_non_center_regressions_ignored():
    input = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 1.0]],
            [[1.0, 1.0], [1.0, 1.0]],
            [[2.0, 2.0], [2.0, 2.0]],
            [[3.0, 3.0], [3.0, 3.0]],
            [[4.0, 4.0], [4.0, 4.0]],
        ]
    )

    target = torch.tensor(
        [
            [[0.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 1.0]],
            [[-1.0, -1.0], [-1.0, 2.0]],
            [[-1.0, -1.0], [-1.0, 3.0]],
            [[-1.0, -1.0], [-1.0, 4.0]],
            [[-1.0, -1.0], [-1.0, 5.0]],
        ]
    )

    criterion = CenterNetLoss(reduction="none")

    cls_loss, reg_loss = criterion(input, target)
    assert (reg_loss[..., 0, 0] == 0).all()
    assert (reg_loss[..., 0, 1] == 0).all()
    assert (reg_loss[..., 1, 1] == 0.5).all()
