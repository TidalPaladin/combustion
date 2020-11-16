#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import torch

from combustion.lightning.metrics import BoxAveragePrecision


class TestBoxAveragePrecision:
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
            ]
        ).float()

        target = torch.tensor(
            [
                [0, 0, 2, 2, 0],
                [2, 2, 4, 4, 1],
                [5, 5, 9, 9, 2],
            ]
        ).float()

        metric = BoxAveragePrecision(iou_threshold=0.5, pos_label=pos_label, true_positive_limit=true_positive_limit)

        value = metric(pred, target)
        assert value == 1.0

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

        metric = BoxAveragePrecision(iou_threshold=0.5, pos_label=pos_label, true_positive_limit=true_positive_limit)

        value = metric(pred, target)
        assert value == 0.5

    @pytest.mark.parametrize(
        "true_positive_limit,iou_threshold,pos_label,expected",
        [
            pytest.param(True, 0.5, None, 0.4379),
            pytest.param(False, 0.5, None, 0.4379),
            pytest.param(True, 0.25, None, 0.6560),
            pytest.param(True, 0.75, None, 0.4071),
            pytest.param(True, 0.5, 0, 0.5833),
            pytest.param(True, 0.5, 2, 0.3333),
        ],
    )
    def test_real_example(self, true_positive_limit, iou_threshold, pos_label, expected):
        torch.random.manual_seed(42)

        # assume we predicted a set of boxes (x1, y1, x2, y2, class)
        pred = torch.tensor(
            [
                [0, 0, 2, 2, 0.5, 0],  # overlaps target[0] and target[1]
                [2, 2, 4, 4, 0.7, 0],  # false positive
                [1, 1, 4, 4, 0.3, 1],  # overlaps target[4] better than V
                [2, 2, 6, 6, 0.7, 1],  # overlaps target[4] worse than ^
                [5, 5, 9, 9, 0.2, 1],  # overlaps target[2]
                [2, 2, 6, 6, 0.7, 2],
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

        metric = BoxAveragePrecision(
            iou_threshold=iou_threshold, pos_label=pos_label, true_positive_limit=true_positive_limit
        )
        value = metric(pred, target)
        expected = torch.tensor(expected)
        assert torch.allclose(value, expected, atol=1e-3)
