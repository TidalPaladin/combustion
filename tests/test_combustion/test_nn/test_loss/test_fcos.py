#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import torch
from torch import Tensor

from combustion.nn import FCOSLoss


class TestFCOSLoss:
    @pytest.mark.parametrize(
        "stride,center_radius,size_target",
        [
            pytest.param(1, None, (10, 10)),
            pytest.param(2, 1, (10, 10)),
            pytest.param(1, 2, (15, 15)),
            pytest.param(1, None, (10, 15)),
        ],
    )
    def test_bbox_to_mask(self, stride, center_radius, size_target):
        bbox = torch.tensor(
            [
                [0, 0, 9, 9],
                [2, 2, 5, 5],
            ]
        )
        result = FCOSLoss.bbox_to_mask(bbox, stride, size_target, center_radius)

        assert isinstance(result, Tensor)
        assert result.shape == torch.Size([bbox.shape[-2], *size_target])

        for box, res in zip(bbox, result):
            h1, w1, h2, w2 = box[1], box[0], box[3], box[2]
            hs1 = h1.floor_divide(stride)
            ws1 = w1.floor_divide(stride)
            hs2 = h2.floor_divide(stride)
            ws2 = w2.floor_divide(stride)

            if center_radius is None:
                pos_region = res[hs1 + 1 : hs2, ws1 + 1 : ws2]
            else:
                x1 = (box[0] + box[2]).floor_divide_(2) - center_radius * stride
                x2 = (box[0] + box[2]).floor_divide_(2) + center_radius * stride
                y1 = (box[1] + box[3]).floor_divide_(2) - center_radius * stride
                y2 = (box[1] + box[3]).floor_divide_(2) + center_radius * stride
                x1.floor_divide_(stride)
                y1.floor_divide_(stride)
                x2.floor_divide_(stride)
                y2.floor_divide_(stride)
                pos_region = res[y1 + 1 : y2, x1 + 1 : x2]

            assert pos_region.all()
            assert res.sum() - pos_region.sum() == 0

    @pytest.mark.parametrize(
        "size_target,stride",
        [
            pytest.param((15, 15), 1),
            pytest.param((10, 10), 2),
            pytest.param((16, 16), 4),
        ],
    )
    def test_create_regression_target(self, size_target, stride):
        bbox = torch.tensor(
            [
                [0, 0, 9, 9],
                [2, 3, 8, 7],
            ]
        ).mul_(stride)
        result = FCOSLoss.create_regression_target(bbox, stride, size_target)

        assert isinstance(result, Tensor)
        assert result.shape == torch.Size([bbox.shape[-2], 4, *size_target])

        for box, res in zip(bbox, result):
            h1, w1, h2, w2 = box[1], box[0], box[3], box[2]
            hs1 = h1.floor_divide(stride)
            ws1 = w1.floor_divide(stride)
            hs2 = h2.floor_divide(stride)
            ws2 = w2.floor_divide(stride)

            pos_region = res[..., hs1 + 1 : hs2, ws1 + 1 : ws2]
            if pos_region.numel():
                assert (pos_region >= 0).all()
                assert pos_region.max() <= box.max()

            def discretize(x):
                return x.floor_divide(stride).mul_(stride)

            discretized_box = box - discretize(box)
            discretized_box[:2].neg_()

            # left
            assert res[0, hs1, ws1] == discretized_box[0], "left target at top left corner"
            assert res[0, hs2, ws1] == discretized_box[0], "left target at bottom left corner"
            assert res[0, hs1, ws2] == discretize(w2 - w1), "left target at top right corner"
            assert res[0, hs2, ws2] == discretize(w2 - w1), "left target at bottom right corner"

            # top
            assert res[1, hs1, ws1] == discretized_box[1], "top target at top left corner"
            assert res[1, hs2, ws1] == discretize(h2 - h1), "top target at bottom left corner"
            assert res[1, hs1, ws2] == discretized_box[1], "top target at top right corner"
            assert res[1, hs2, ws2] == discretize(h2 - h1), "top target at bottom right corner"

            # right
            assert res[2, hs1, ws1] == w2 - w1, "right target at top left corner"
            assert res[2, hs2, ws1] == w2 - w1, "right target at bottom left corner"
            assert res[2, hs1, ws2] == discretized_box[2], "right target at top right corner"
            assert res[2, hs2, ws2] == discretized_box[2], "right target at bottom right corner"

            # bottom
            assert res[3, hs1, ws1] == h2 - h1, "right target at top left corner"
            assert res[3, hs2, ws1] == discretized_box[3], "right target at bottom left corner"
            assert res[3, hs1, ws2] == h2 - h1, "right target at top right corner"
            assert res[3, hs2, ws2] == discretized_box[3], "right target at bottom right corner"

    @pytest.mark.parametrize(
        "stride,center_radius,size_target",
        [
            pytest.param(1, None, (10, 10)),
            pytest.param(1, 1, (15, 15)),
        ],
    )
    def test_create_classification_target(self, stride, center_radius, size_target):
        bbox = torch.tensor(
            [
                [0, 0, 9, 9],
                [3, 4, 8, 6],
                [4, 4, 6, 6],
            ]
        )
        cls = torch.tensor([0, 0, 1]).unsqueeze_(-1)
        mask = FCOSLoss.bbox_to_mask(bbox, stride, size_target, center_radius)
        num_classes = 2

        result = FCOSLoss.create_classification_target(bbox, cls, mask, num_classes, size_target)

        assert isinstance(result, Tensor)
        assert result.shape == torch.Size([num_classes, *size_target])

    @pytest.mark.parametrize(
        "stride,center_radius,size_target",
        [
            pytest.param(1, None, (10, 10)),
            pytest.param(1, 1, (15, 15)),
        ],
    )
    def test_create_target_for_level(self, stride, center_radius, size_target):
        bbox = torch.tensor(
            [
                [0, 0, 9, 9],
                [3, 4, 8, 6],
                [4, 4, 6, 6],
            ]
        )
        cls = torch.tensor([0, 0, 1]).unsqueeze_(-1)
        num_classes = 2

        cls, reg, centerness = FCOSLoss.create_target_for_level(
            bbox, cls, num_classes, stride, size_target, [-1, 64], center_radius
        )

        assert cls.shape == torch.Size([num_classes, *size_target])
        assert reg.shape == torch.Size([4, *size_target])
        assert centerness.shape == torch.Size([1, *size_target])

        # TODO expand on this test

        assert centerness.max() <= 1.0
        assert ((centerness >= 0) | (centerness == -1)).all()

    def test_compute_loss(self):
        target_bbox = torch.tensor(
            [
                [0, 0, 9, 9],
                [3, 4, 8, 6],
                [4, 4, 6, 6],
                [32, 32, 88, 88],
            ]
        )
        target_cls = torch.tensor([0, 0, 1, 0]).unsqueeze_(-1)

        num_classes = 2
        strides = [8, 16, 32, 64, 128]
        base_size = 512
        sizes = [(base_size // stride,) * 2 for stride in strides]

        pred_cls = [torch.rand(num_classes, *size, requires_grad=True) for size in sizes]
        pred_reg = [torch.rand(4, *size, requires_grad=True).mul(512).round() for size in sizes]
        pred_centerness = [torch.rand(1, *size, requires_grad=True) for size in sizes]

        criterion = FCOSLoss(strides, num_classes)
        cls_loss, reg_loss, centerness_loss = criterion.compute_from_box_target(
            pred_cls, pred_reg, pred_centerness, target_bbox, target_cls
        )

        assert isinstance(cls_loss, Tensor)
        assert isinstance(reg_loss, Tensor)
        assert isinstance(centerness_loss, Tensor)

        assert cls_loss.numel() == 1
        assert reg_loss.numel() == 1
        assert centerness_loss.numel() == 1

        loss = cls_loss + reg_loss + centerness_loss
        loss.backward()

    def test_call(self):
        target_bbox = torch.tensor(
            [
                [
                    [0, 0, 9, 9],
                    [3, 4, 8, 6],
                    [-1, -1, -1, -1],
                ],
                [
                    [32, 32, 88, 88],
                    [-1, -1, -1, -1],
                    [-1, -1, -1, -1],
                ],
                [
                    [-1, -1, -1, -1],
                    [-1, -1, -1, -1],
                    [-1, -1, -1, -1],
                ],
            ]
        )

        target_cls = torch.tensor(
            [
                [0, 1, -1],
                [0, -1, -1],
                [-1, -1, -1],
            ]
        ).unsqueeze_(-1)

        batch_size = target_bbox.shape[0]
        num_classes = 2
        strides = [8, 16, 32, 64, 128]
        base_size = 512
        sizes = [(base_size // stride,) * 2 for stride in strides]

        pred_cls = [torch.rand(batch_size, num_classes, *size, requires_grad=True) for size in sizes]
        pred_reg = [torch.rand(batch_size, 4, *size, requires_grad=True).mul(512).round() for size in sizes]
        pred_centerness = [torch.rand(batch_size, 1, *size, requires_grad=True) for size in sizes]

        criterion = FCOSLoss(strides, num_classes)
        cls_loss, reg_loss, centerness_loss = criterion(pred_cls, pred_reg, pred_centerness, target_bbox, target_cls)

        assert isinstance(cls_loss, Tensor)
        assert isinstance(reg_loss, Tensor)
        assert isinstance(centerness_loss, Tensor)

        assert cls_loss.numel() == 1
        assert reg_loss.numel() == 1
        assert centerness_loss.numel() == 1

        loss = cls_loss + reg_loss + centerness_loss
        loss.backward()
