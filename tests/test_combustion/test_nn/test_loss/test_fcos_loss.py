#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from typing import Optional

import pytest
import torch
from torch import Tensor

from combustion.nn import FCOSLoss
from combustion.util import alpha_blend, apply_colormap
from combustion.vision import visualize_bbox


class TestFCOSLoss:
    @pytest.mark.parametrize(
        "stride,center_radius,size_target",
        [
            pytest.param(1, None, (10, 10)),
            pytest.param(2, None, (5, 5)),
            pytest.param(2, 1, (5, 5)),
            pytest.param(2, 10, (10, 10)),
            pytest.param(1, 2, (15, 15)),
            pytest.param(1, None, (10, 15)),
            pytest.param(1, None, (10, 15)),
        ],
    )
    def test_bbox_to_mask(self, stride, center_radius, size_target):
        bbox = torch.tensor(
            [
                [0, 0, 9, 9],
                [2, 2, 5, 5],
                [1, 1, 2, 2],
            ]
        )
        result = FCOSLoss.bbox_to_mask(bbox, stride, size_target, center_radius)

        assert isinstance(result, Tensor)
        assert result.shape == torch.Size([bbox.shape[-2], *size_target])

        for box, res in zip(bbox, result):
            h1, w1, h2, w2 = box[1], box[0], box[3], box[2]

            center_x = (box[0] + box[2]).true_divide(2)
            center_y = (box[1] + box[3]).true_divide(2)
            radius_x = (box[2] - box[0]).true_divide(2)
            radius_y = (box[3] - box[1]).true_divide(2)

            if center_radius is not None:
                x1 = center_x - center_radius * stride
                x2 = center_x + center_radius * stride
                y1 = center_y - center_radius * stride
                y2 = center_y + center_radius * stride
            else:
                x1 = center_x - radius_x
                x2 = center_x + radius_x
                y1 = center_y - radius_y
                y2 = center_y + radius_y

            x1.clamp_min_(center_x - radius_x)
            x2.clamp_max_(center_x + radius_x)
            y1.clamp_min_(center_y - radius_y)
            y2.clamp_max_(center_y + radius_y)

            h = torch.arange(res.shape[-2], dtype=torch.float, device=box.device)
            w = torch.arange(res.shape[-1], dtype=torch.float, device=box.device)

            mesh = torch.stack(torch.meshgrid(h, w), 0).mul_(stride).add_(stride / 2)
            lower_bound = torch.stack([x1, y1]).view(2, 1, 1)
            upper_bound = torch.stack([x2, y2]).view(2, 1, 1)
            mask = (mesh >= lower_bound).logical_and_(mesh <= upper_bound).all(dim=-3)
            pos_region = res[mask]

            assert res.any()
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

            pos_region = res[..., hs1:hs2, ws1:ws2]
            if pos_region.numel():
                assert (pos_region >= 0).all()
                assert pos_region.max() <= box.max()

            def discretize(x):
                return x.floor_divide(stride).mul_(stride).add_(stride // 2)

            # left
            assert res[0, hs1, ws1] == stride // 2, "left target at top left corner"
            assert res[0, hs2, ws1] == stride // 2, "left target at bottom left corner"
            assert res[0, hs1, ws2] == discretize(w2 - w1), "left target at top right corner"
            assert res[0, hs2, ws2] == discretize(w2 - w1), "left target at bottom right corner"

            # top
            assert res[1, hs1, ws1] == stride // 2, "top target at top left corner"
            assert res[1, hs2, ws1] == discretize(h2 - h1), "top target at bottom left corner"
            assert res[1, hs1, ws2] == stride // 2, "top target at top right corner"
            assert res[1, hs2, ws2] == discretize(h2 - h1), "top target at bottom right corner"

            # right
            assert res[2, hs1, ws1] == w2 - w1 - stride // 2, "right target at top left corner"
            assert res[2, hs2, ws1] == w2 - w1 - stride // 2, "right target at bottom left corner"
            assert res[2, hs1, ws2] == stride // 2, "right target at top right corner"
            assert res[2, hs2, ws2] == stride // 2, "right target at bottom right corner"

            # bottom
            assert res[3, hs1, ws1] == h2 - h1 - stride // 2, "right target at top left corner"
            assert res[3, hs2, ws1] == stride // 2, "right target at bottom left corner"
            assert res[3, hs1, ws2] == h2 - h1 - stride // 2, "right target at top right corner"
            assert res[3, hs2, ws2] == stride // 2, "right target at bottom right corner"

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

    @pytest.mark.parametrize(
        "stride,center_radius,size_target",
        [
            pytest.param(1, None, (10, 10)),
            pytest.param(1, 1, (15, 15)),
        ],
    )
    def test_create_targets(self, stride, center_radius, size_target):
        num_classes = 2
        target_bbox = torch.randint(0, 100, (2, 10, 4))
        target_cls = torch.randint(0, num_classes, (2, 10, 1))

        strides = [8, 16, 32, 64, 128]
        base_size = 512
        sizes = [(base_size // stride,) * 2 for stride in strides]

        criterion = FCOSLoss(strides, num_classes)
        criterion.create_targets(target_bbox, target_cls, sizes)

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
        assert not loss.isnan().any()
        loss.backward()

    # Set this to a directory to write out some sample images from test cases
    # DEST: Optional[str] = None
    DEST: Optional[str] = "/home/tidal"

    def save(self, path, result):
        import matplotlib.pyplot as plt

        plt.imsave(path, result.permute(1, 2, 0).cpu().numpy())

    def blend_and_save(self, path, src, dest):
        src = apply_colormap(src)[..., :3, :, :]
        src = torch.nn.functional.interpolate(src, dest.shape[-2:])
        _ = alpha_blend(src, dest)[0].squeeze_(0)
        self.save(path, _)

    @pytest.mark.parametrize(
        "center_radius",
        [
            pytest.param(None),
            pytest.param(1),
            pytest.param(3),
            pytest.param(20),
        ],
    )
    def test_save_output(self, center_radius):
        image_size = 512
        num_classes = 2
        target_bbox = torch.tensor(
            [
                [140, 140, 144, 144],
                [10, 10, 128, 128],
                [32, 64, 128, 256],
                [250, 10, 250 + 31, 10 + 19],
                [256, 256, 400, 512],
            ]
        )
        img = torch.zeros(1, image_size, image_size)
        target_cls = torch.tensor([1, 0, 1, 1, 0]).unsqueeze_(-1)

        strides = [8, 16, 32, 64, 128]
        sizes = [(image_size // stride,) * 2 for stride in strides]

        criterion = FCOSLoss(strides, num_classes, radius=center_radius)
        targets = criterion.create_targets(target_bbox, target_cls, sizes)
        cls_targets = [t[0] for t in targets]
        reg_targets = [t[1] for t in targets]
        centerness_targets = [t[2] for t in targets]

        reg_targets = [torch.linalg.norm(x.float().clamp_min(0), dim=-3, keepdim=True) for x in reg_targets]
        reg_targets = [x.div(x.amax(dim=(-1, -2, -3), keepdim=True).clamp_min_(1)) for x in reg_targets]
        centerness_targets = [x.clamp_min_(0) for x in centerness_targets]

        img_with_box = visualize_bbox(img, target_bbox, target_cls)[None]

        subpath = os.path.join(self.DEST, "fcos_targets")
        if not os.path.exists(subpath):
            os.makedirs(subpath)

        subpath = os.path.join(subpath, f"radius_{center_radius}")
        if not os.path.exists(subpath):
            os.makedirs(subpath)

        for level in range(len(strides)):
            image_path = os.path.join(subpath)
            c = cls_targets[level][None]
            r = reg_targets[level][None]
            cent = centerness_targets[level][None]

            filename = os.path.join(image_path, f"reg_level_{level}.png")
            self.blend_and_save(filename, r, img_with_box)

            filename = os.path.join(image_path, f"centerness_level_{level}.png")
            self.blend_and_save(filename, cent, img_with_box)

            for cls_idx in range(c.shape[1]):
                filename = os.path.join(image_path, f"cls_{cls_idx}_level_{level}.png")
                self.blend_and_save(filename, c[..., cls_idx, :, :][None], img_with_box)
