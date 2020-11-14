#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch

from combustion.lightning.metrics import AUROC


def test_compute_single_step():
    pred = torch.tensor([0.1, 0.7, 0.3, 0.4, 0.9])
    target = torch.tensor([0, 1, 1, 0, 1])
    metric = AUROC()
    score = metric(pred, target)

    expected = torch.tensor(0.8333)  # computed via sklearn
    assert torch.allclose(score, expected, atol=1e-4)


def test_compute_multi_step():
    pred = torch.tensor([0.1, 0.7, 0.3, 0.4, 0.9])
    target = torch.tensor([0, 1, 1, 0, 1])
    metric = AUROC(compute_on_step=False)
    metric.update(pred, target)

    scale = torch.tensor([0.9, 1.1, 1.0, 0.8, 1.3])
    metric.update(pred.mul(3 * scale).sigmoid(), target)
    score = metric.compute()

    expected = torch.tensor(0.79166)  # computed via sklearn
    assert torch.allclose(score, expected, atol=1e-4)


def test_reset_after_compute():
    pred = torch.tensor([0.1, 0.7, 0.3, 0.4, 0.9])
    target = torch.tensor([0, 1, 1, 0, 1])
    metric = AUROC(compute_on_step=False)
    metric.update(pred, target)
    score = metric.compute()

    expected = torch.tensor(0.8333)  # computed via sklearn
    assert torch.allclose(score, expected, atol=1e-4)

    metric.update(pred, target)
    score = metric.compute()
    assert torch.allclose(score, expected, atol=1e-4)
