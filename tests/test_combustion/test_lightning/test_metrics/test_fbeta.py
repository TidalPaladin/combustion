#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch

from combustion.lightning.metrics import Fbeta


def test_compute_single_step():
    pred = torch.tensor([0, 1, 2, 1, 0, 0])
    target = torch.tensor([0, 0, 1, 2, 1, 0])

    metric = Fbeta(beta=1.0, average="weighted", num_classes=3)
    score = metric(pred, target)

    expected = torch.tensor(0.3333)  # computed via sklearn
    assert torch.allclose(score, expected, atol=1e-4)


def test_compute_multi_step():
    pred = torch.tensor([0, 1, 2, 1, 0, 0])
    target = torch.tensor([0, 0, 1, 2, 1, 0])
    metric = Fbeta(beta=1.0, average="weighted", num_classes=3)
    metric.update(pred, target)

    pred2 = torch.tensor([0, 1, 2, 2, 1, 0])
    metric.update(pred2, target)
    score = metric.compute()

    expected = torch.tensor(0.513636)  # computed via sklearn
    assert torch.allclose(score, expected, atol=1e-4)


def test_reset_after_compute():
    pred = torch.tensor([0, 1, 2, 1, 0, 0])
    target = torch.tensor([0, 0, 1, 2, 1, 0])
    metric = Fbeta(beta=1.0, average="weighted", num_classes=3)
    metric.update(pred, target)
    score = metric.compute()

    expected = torch.tensor(0.3333)  # computed via sklearn
    assert torch.allclose(score, expected, atol=1e-4)

    metric.update(pred, target)
    score = metric.compute()
    assert torch.allclose(score, expected, atol=1e-4)
