#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch

from combustion.vision import ConfusionMatrixIoU


def test_iou_assign():
    pred_box = torch.tensor([[1, 1, 4, 4], [2, 2, 5, 5], [1, 1, 2, 2],]).float()
    true_box = torch.tensor(
        [[1, 1, 20, 20], [2, 2, 4.5, 4.5], [2, 2, 4, 4], [0, 0, 2, 2], [3, 3, 5, 5], [0.9, 0.9, 1.9, 1.9],]
    ).float()
    pred_cls = torch.tensor([0, 1, 1]).unsqueeze(-1)
    true_cls = torch.tensor([1, 1, 0, 0, 1, 0]).unsqueeze(-1)

    layer = ConfusionMatrixIoU()
    tp, fn = layer(pred_box, pred_cls, true_box, true_cls)
    assert torch.allclose(tp.long(), torch.tensor([0, 1, 0]))
    assert torch.allclose(fn.long(), torch.tensor([1, 0, 1, 1, 1, 1]))
