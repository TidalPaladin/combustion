#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import torch

from combustion.vision import BinaryLabelIoU, ConfusionMatrixIoU


class TestConfusionMatrixIoU:
    def test_basic(self):
        pred_box = torch.tensor(
            [
                [1, 1, 4, 4],
                [2, 2, 5, 5],
                [1, 1, 2, 2],
            ]
        ).float()
        true_box = torch.tensor(
            [
                [1, 1, 20, 20],
                [2, 2, 4.5, 4.5],
                [2, 2, 4, 4],
                [0, 0, 2, 2],
                [3, 3, 5, 5],
                [0.9, 0.9, 1.9, 1.9],
            ]
        ).float()
        pred_cls = torch.tensor([0, 1, 1]).unsqueeze(-1)
        true_cls = torch.tensor([1, 1, 0, 0, 1, 0]).unsqueeze(-1)

        layer = ConfusionMatrixIoU()
        tp, fn = layer(pred_box, pred_cls, true_box, true_cls)
        assert torch.allclose(tp.long(), torch.tensor([0, 1, 0]))
        assert torch.allclose(fn.long(), torch.tensor([1, 0, 1, 1, 1, 1]))

    @pytest.mark.parametrize("true_positive_limit", [True, False])
    def test_excludes_duplicate_tp(self, true_positive_limit):
        pred_box = torch.tensor(
            [
                [1, 1, 4, 4],
                [2, 2, 5, 5],
                [1, 1, 2, 2],
                [2, 2, 4.1, 4.1],
                [2, 2, 4.2, 4.2],
            ],
        ).float()
        true_box = torch.tensor(
            [
                [1, 1, 20, 20],
                [2, 2, 4.5, 4.5],
                [2, 2, 4, 4],
                [0, 0, 2, 2],
                [3, 3, 5, 5],
                [0.9, 0.9, 1.9, 1.9],
            ]
        ).float()
        pred_cls = torch.tensor([0, 1, 1, 0, 0]).unsqueeze(-1)
        true_cls = torch.tensor([1, 1, 0, 0, 1, 0]).unsqueeze(-1)

        layer = ConfusionMatrixIoU(true_positive_limit=true_positive_limit)
        tp, fn = layer(pred_box, pred_cls, true_box, true_cls)

        if true_positive_limit:
            assert torch.allclose(tp.long(), torch.tensor([0, 1, 0, 0, 1]))
        else:
            assert torch.allclose(tp.long(), torch.tensor([0, 1, 0, 1, 1]))
        assert torch.allclose(fn.long(), torch.tensor([1, 0, 0, 1, 1, 1]))


class TestBinaryLabelIoU:
    @pytest.mark.parametrize("true_positive_limit", [True, False])
    def test_result(self, true_positive_limit):
        pred_box = torch.tensor(
            [
                [1, 1, 4, 4],
                [2, 2, 5, 5],
                [1, 1, 2, 2],
                [2, 2, 4.1, 4.1],
                [2, 2, 4.2, 4.2],
            ],
        ).float()
        true_box = torch.tensor(
            [
                [1, 1, 20, 20],
                [2, 2, 4.5, 4.5],
                [2, 2, 4, 4],
                [0, 0, 2, 2],
                [3, 3, 5, 5],
                [0.9, 0.9, 1.9, 1.9],
            ]
        ).float()
        pred_cls = torch.tensor([0, 1, 1, 0, 0]).unsqueeze(-1)
        pred_score = torch.tensor([0.5, 0.25, 0.75, 0.001, 0.2]).unsqueeze(-1)
        true_cls = torch.tensor([1, 1, 0, 0, 1, 0]).unsqueeze(-1)

        layer = BinaryLabelIoU(true_positive_limit=true_positive_limit)
        pred, true = layer(pred_box, pred_score, pred_cls, true_box, true_cls)

        if true_positive_limit:
            expected_pred = torch.tensor([0.5000, 0.2500, 0.7500, 0.0010, 0.2000, 0.0000, 0.0000, 0.0000, 0.0000])
            expected_true = torch.tensor([0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        else:
            expected_pred = torch.tensor([0.5000, 0.2500, 0.7500, 0.0010, 0.2000, 0.0000, 0.0000, 0.0000, 0.0000])
            expected_true = torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        assert torch.allclose(expected_pred, pred)
        assert torch.allclose(expected_true, true)
