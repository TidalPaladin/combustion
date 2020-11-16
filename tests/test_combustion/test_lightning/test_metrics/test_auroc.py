#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import torch

from combustion.lightning.metrics import AUROC, BoxAUROC


class TestAUROC:
    def test_compute_single_step(self):
        pred = torch.tensor([0.1, 0.7, 0.3, 0.4, 0.9])
        target = torch.tensor([0, 1, 1, 0, 1])
        metric = AUROC()
        score = metric(pred, target)

        expected = torch.tensor(0.8333)  # computed via sklearn
        assert torch.allclose(score, expected, atol=1e-4)

    def test_compute_multi_step(self):
        pred = torch.tensor([0.1, 0.7, 0.3, 0.4, 0.9])
        target = torch.tensor([0, 1, 1, 0, 1])
        metric = AUROC(compute_on_step=False)
        metric.update(pred, target)

        scale = torch.tensor([0.9, 1.1, 1.0, 0.8, 1.3])
        metric.update(pred.mul(3 * scale).sigmoid(), target)
        score = metric.compute()

        expected = torch.tensor(0.79166)  # computed via sklearn
        assert torch.allclose(score, expected, atol=1e-4)

    def test_reset_after_compute(self):
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


class TestBoxAUROC:
    @pytest.mark.parametrize("true_positive_limit", [True, False])
    @pytest.mark.parametrize("pos_label", [None, 0, 1, 2])
    def test_all_correct(self, true_positive_limit, pos_label):
        torch.random.manual_seed(42)

        # assume we predicted a set of boxes (x1, y1, x2, y2, class)
        pred = torch.tensor(
            [
                [0, 0, 2, 2, 1.0, 0],
                [2, 2, 4, 4, 1.0, 1],
                [5, 5, 9, 9, 1.0, 2],
                [32, 32, 64, 64, 0.0, 2],
            ]
        ).float()

        target = torch.tensor(
            [
                [0, 0, 2, 2, 0],
                [2, 2, 4, 4, 1],
                [5, 5, 9, 9, 2],
            ]
        ).float()

        metric = BoxAUROC(iou_threshold=0.5, pos_label=pos_label, true_positive_limit=true_positive_limit)

        value = metric(pred, target)
        if pos_label == 2.0 or pos_label is None:
            assert value == 1.0
        else:
            # nan when no positive targets exist
            assert value.isnan().all()

    @pytest.mark.parametrize("true_positive_limit", [True, False])
    @pytest.mark.parametrize("pos_label", [None, 0, 1, 2])
    def test_all_wrong(self, true_positive_limit, pos_label):
        torch.random.manual_seed(42)

        # assume we predicted a set of boxes (x1, y1, x2, y2, class)
        pred = torch.tensor(
            [
                [0, 0, 2, 2, 1.0, 0],
                [2, 2, 4, 4, 1.0, 1],
                [5, 5, 9, 9, 1.0, 2],
            ]
        ).float()

        target = torch.tensor(
            [
                [0, 0, 2, 2, 1],
                [2, 2, 4, 4, 2],
                [5, 5, 9, 9, 0],
            ]
        ).float()

        metric = BoxAUROC(iou_threshold=0.5, pos_label=pos_label, true_positive_limit=true_positive_limit)

        value = metric(pred, target)
        assert value == 0.0

    @pytest.mark.parametrize(
        "true_positive_limit,iou_threshold,pos_label,expected",
        [
            pytest.param(True, 0.5, None, 0.5500),
            pytest.param(False, 0.5, None, 0.550),
            pytest.param(True, 0.25, None, 0.4667),
            pytest.param(True, 0.75, None, 0.32),
            pytest.param(True, 0.5, 0, 0.5000),
            pytest.param(True, 0.5, 2, 0.0),
        ],
    )
    def test_real_example(self, true_positive_limit, iou_threshold, pos_label, expected):
        torch.random.manual_seed(42)

        # assume we predicted a set of boxes (x1, y1, x2, y2, class)
        pred = torch.tensor(
            [
                [0, 0, 2, 2, 0.9, 0],  # overlaps target[0] and target[1]
                [2, 2, 4, 4, 0.2, 0],  # false positive
                [1, 1, 4, 4, 0.8, 1],  # overlaps target[4] better than V
                [2, 2, 6, 6, 0.7, 1],  # overlaps target[4] worse than ^
                [5, 5, 9, 9, 0.5, 1],  # overlaps target[2]
                [2, 2, 6, 6, 0.1, 2],
                [5, 5, 9, 9, 0.2, 2],
            ]
        ).float()

        target = torch.tensor(
            [
                [0, 0, 2.1, 2.1, 0],  # tp pred_bbox[0]
                [0, 0, 2, 2, 0],  # tp pred_bbox[0]
                [3, 3, 6, 6, 0],  # false negative
                [5, 5, 10, 9, 1],  # tp pred_bbox[4]
                [1, 1, 4.9, 4.9, 1],  # tp pred_bbox[3]
                [5, 5, 10, 9, 1],  # tp pred_bbox[4]
                [1, 1, 4.9, 4.9, 2],  # tp pred_bbox[3]
            ]
        ).float()

        metric = BoxAUROC(iou_threshold=iou_threshold, pos_label=pos_label, true_positive_limit=true_positive_limit)
        value = metric(pred, target)
        expected = torch.tensor(expected)
        assert torch.allclose(value, expected, atol=1e-3)
