#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gc
import os
from typing import Optional

import pytest
import torch
from torch import Tensor

from combustion.nn import FCOSDecoder, FCOSLoss
from combustion.testing import TorchScriptTestMixin
from combustion.vision import visualize_bbox


class TestFCOSDecoder(TorchScriptTestMixin):
    @pytest.fixture
    def in_channels(self):
        return 10

    @pytest.fixture
    def num_classes(self):
        return 3

    @pytest.fixture
    def strides(self):
        return [1, 2, 4]

    @pytest.fixture
    def num_convs(self):
        return 2

    @pytest.fixture
    def data(self, in_channels, strides):
        base_size = 64
        data = []
        for s in strides:
            size = base_size // s
            _ = torch.rand(1, in_channels, size, size)
            data.append(_)
        return data

    @pytest.fixture
    def model_type(self):
        return FCOSDecoder

    @pytest.fixture
    def model(self, model_type, in_channels, num_classes, strides, num_convs):
        model = model_type(in_channels, num_classes, num_convs, strides=strides)
        yield model
        del model
        gc.collect()

    def test_forward(self, model, data, num_classes, strides):

        cls_pred, reg_pred, centering = model(data)
        for x in (cls_pred, reg_pred, centering):
            assert isinstance(x, list)
            assert all([isinstance(y, Tensor) for y in x])

        for i, (cls, reg, cen) in enumerate(zip(cls_pred, reg_pred, centering)):
            assert cls.shape[1] == num_classes
            assert reg.shape[1] == 4
            assert cen.shape[1] == 1

            assert cls.shape[2:] == data[i].shape[2:]
            assert reg.shape[2:] == data[i].shape[2:]
            assert cls.shape[2:] == data[i].shape[2:]

    def test_postprocess_max_boxes(self, model_type):
        num_classes = 2
        strides = [8, 16, 32, 64, 128]
        base_size = 512
        sizes = [(base_size // stride,) * 2 for stride in strides]

        torch.random.manual_seed(42)
        pred_cls = [torch.rand(2, num_classes, *size, requires_grad=True) for size in sizes]
        pred_reg = [torch.rand(2, 4, *size, requires_grad=True).mul(512).round() for size in sizes]
        pred_centerness = [torch.rand(2, 1, *size, requires_grad=True) for size in sizes]

        max_boxes = 10
        boxes = model_type.postprocess(pred_cls, pred_reg, pred_centerness, strides, max_boxes=max_boxes)
        assert isinstance(boxes, Tensor)
        assert boxes.shape[-1] == 6
        assert boxes.shape[-2] == max_boxes
        assert (boxes[..., -1] >= 0).all()

    def test_postprocess2(self, model_type):
        num_classes = 2
        strides = [8, 16, 32, 64, 128]
        base_size = 512
        sizes = [(base_size // stride,) * 2 for stride in strides]

        pred_cls = [torch.zeros(2, num_classes, *size, requires_grad=True) for size in sizes]
        pred_reg = [torch.ones(2, 4, *size, requires_grad=True).mul(10).round() for size in sizes]
        pred_centerness = [torch.ones(2, 1, *size, requires_grad=True).mul(0.5) for size in sizes]

        pred_cls[-1][0, 0, 1, 1] = 0.91
        pred_cls[-3][0, 1, 5, 5] = 0.92
        pred_cls[-2][1, 1, 3, 3] = 0.93
        pred_cls[0][1, 1, 2, 2] = 0.94
        pred_cls[2][0, 1, 10, 10] = 0.95

        boxes = model_type.postprocess(pred_cls, pred_reg, pred_centerness, strides, use_raw_score=True)
        assert isinstance(boxes, Tensor)

    def test_postprocess_no_positives(self, model_type):
        num_classes = 2
        strides = [8, 16, 32, 64, 128]
        base_size = 512
        sizes = [(base_size // stride,) * 2 for stride in strides]

        pred_cls = [torch.zeros(2, num_classes, *size, requires_grad=True) for size in sizes]
        pred_reg = [torch.ones(2, 4, *size, requires_grad=True).mul(10).round() for size in sizes]
        pred_centerness = [torch.ones(2, 1, *size, requires_grad=True).mul(0.5) for size in sizes]

        boxes = model_type.postprocess(pred_cls, pred_reg, pred_centerness, strides, use_raw_score=True)
        assert isinstance(boxes, Tensor)
        assert boxes.numel() == 0

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

    def test_save_output(self, model_type):
        torch.random.manual_seed(42)
        image_size = 512
        num_classes = 2
        batch_size = 3
        center_radius = 2
        target_bbox = (
            torch.tensor(
                [
                    [10, 10, 128, 128],
                    [12, 12, 130, 130],
                    [32, 64, 128, 256],
                    [256, 256, 400, 512],
                ]
            )
            .unsqueeze_(0)
            .repeat(batch_size, 1, 1)
        )
        img = torch.zeros(4, 1, image_size, image_size)
        target_cls = torch.tensor([0, 0, 1, 1]).unsqueeze_(-1).repeat(batch_size, 1, 1)

        strides = [8, 16, 32, 64, 128]
        sizes = [(image_size // stride,) * 2 for stride in strides]

        criterion = FCOSLoss(strides, num_classes, radius=center_radius)
        cls_targets, reg_targets, centerness_targets = criterion.create_targets(target_bbox, target_cls, sizes)

        final_pred = model_type.postprocess(cls_targets, reg_targets, centerness_targets, strides, threshold=0.5)
        final_boxes = final_pred[..., :4]
        final_scores = final_pred[..., 4:5]
        final_cls = final_pred[..., 5:]

        img_with_box = visualize_bbox(img, final_boxes, final_scores, final_cls)

        subpath = os.path.join(self.DEST, "fcos_targets")
        if not os.path.exists(subpath):
            os.makedirs(subpath)

        for i, item in enumerate(img_with_box):
            filename = os.path.join(subpath, f"created_targets_{i}.png")
            self.save(filename, item)

    def test_nms(self, model_type):
        torch.random.manual_seed(42)
        image_size = 512
        num_classes = 1
        batch_size = 2
        strides = [8, 16, 32, 64, 128]
        sizes = [(image_size // stride,) * 2 for stride in strides]

        img = torch.zeros(batch_size, 1, image_size, image_size)
        pred_cls = [torch.rand(batch_size, num_classes, *size).sub_(0.45).clamp_min_(0) for size in sizes]
        pred_cls[0].clamp_min(0.111)
        pred_reg = [torch.rand(batch_size, 4, *size).mul_(image_size / 4).round_().clamp_min_(24) for size in sizes]
        pred_centerness = [torch.rand(batch_size, 1, *size) for size in sizes]

        final_pred = model_type.postprocess(
            pred_cls, pred_reg, pred_centerness, strides, threshold=0.5, nms_threshold=None
        )
        final_boxes = final_pred[..., :4]
        final_scores = final_pred[..., 4:5]
        final_cls = final_pred[..., 5:]

        img_with_box = visualize_bbox(img, final_boxes, final_scores, final_cls)

        subpath = os.path.join(self.DEST, "fcos_targets")
        if not os.path.exists(subpath):
            os.makedirs(subpath)

        for i, item in enumerate(img_with_box):
            filename = os.path.join(subpath, f"created_targets_no_nms_{i}.png")
            self.save(filename, item)

        final_pred = model_type.postprocess(
            pred_cls, pred_reg, pred_centerness, strides, threshold=0.5, nms_threshold=0.01
        )
        final_boxes = final_pred[..., :4]
        final_scores = final_pred[..., 4:5]
        final_cls = final_pred[..., 5:]

        img_with_box = visualize_bbox(img, final_boxes, final_cls, final_scores)

        subpath = os.path.join(self.DEST, "fcos_targets")
        if not os.path.exists(subpath):
            os.makedirs(subpath)

        for i, item in enumerate(img_with_box):
            filename = os.path.join(subpath, f"created_targets_nms_{i}.png")
            self.save(filename, item)

    def test_batched_nms(self, model_type):
        torch.random.manual_seed(42)
        image_size = 512
        num_classes = 1
        batch_size = 2
        strides = [8, 16, 32, 64, 128]
        sizes = [(image_size // stride,) * 2 for stride in strides]

        torch.zeros(batch_size, 1, image_size, image_size)
        pred_cls = [torch.rand(batch_size, num_classes, *size).sub_(0.45).clamp_min_(0) for size in sizes]
        pred_reg = [torch.rand(batch_size, 4, *size).mul_(image_size / 4).round_().clamp_min_(24) for size in sizes]
        pred_centerness = [torch.rand(batch_size, 1, *size) for size in sizes]

        batched = model_type.postprocess(pred_cls, pred_reg, pred_centerness, strides, threshold=0.5, nms_threshold=0.3)

        pred_cls = [x[0:1] for x in pred_cls]
        pred_reg = [x[0:1] for x in pred_reg]
        pred_centerness = [x[0:1] for x in pred_centerness]
        unbatched = model_type.postprocess(
            pred_cls, pred_reg, pred_centerness, strides, threshold=0.5, nms_threshold=0.3
        )

        unbatched.squeeze_(0)
        batched = batched[0]
        batched = batched[(batched != -1).all(dim=-1)]
        assert batched.shape == unbatched.shape
        assert torch.allclose(batched, unbatched)
