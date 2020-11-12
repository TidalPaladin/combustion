#!/usr/bin/env python
# -*- coding: utf-8 -*-
import gc
import os

import pytest
import torch
from torch import Tensor
from typing import Optional

from combustion.models import EfficientDetFCOS
from combustion.nn import MobileNetBlockConfig, FCOSLoss
from combustion.testing import TorchScriptTestMixin, TorchScriptTraceTestMixin
from combustion.vision import visualize_bbox


class TestEfficientDetFCOS(TorchScriptTestMixin, TorchScriptTraceTestMixin):
    @pytest.fixture
    def model_type(self):
        return EfficientDetFCOS

    @pytest.fixture
    def data(self):
        return torch.rand(1, 3, 512, 512)

    @pytest.fixture
    def num_classes(self):
        return 3

    @pytest.fixture
    def model(self, num_classes, model_type):
        block1 = MobileNetBlockConfig(4, 8, 3, num_repeats=2, stride=2)
        block2 = MobileNetBlockConfig(8, 16, 3, num_repeats=1, stride=2)
        blocks = [block1, block2]
        model = model_type(num_classes, blocks, [1, 2, 3, 4])
        yield model
        del model
        gc.collect()

    def test_construct(self, model_type, num_classes):
        block1 = MobileNetBlockConfig(4, 8, 3, num_repeats=2)
        block2 = MobileNetBlockConfig(8, 16, 3, num_repeats=1)
        blocks = [block1, block2]
        model_type(num_classes, blocks, [1, 2])
        gc.collect()

    def test_forward(self, model, data, num_classes):
        cls_pred, reg_pred, centering = model(data)
        for x in (cls_pred, reg_pred, centering):
            assert isinstance(x, list)
            assert all([isinstance(y, Tensor) for y in x])

        batch_size = 1

        for i, (cls, reg, cen) in enumerate(zip(cls_pred, reg_pred, centering)):
            assert cls.shape[1] == num_classes
            assert reg.shape[1] == 4
            assert cen.shape[1] == 1

            expected_size = torch.Size([x // (2 ** (i + 2)) for x in data.shape[2:]])
            assert cls.shape[2:] == expected_size
            assert reg.shape[2:] == expected_size
            assert cen.shape[2:] == expected_size
            assert cls.shape[0] == batch_size
            assert reg.shape[0] == batch_size
            assert cen.shape[0] == batch_size

    def test_backward(self, model, data):
        output = model(data)
        scalar = sum([t.sum() for f in output for t in f])
        scalar.backward()

    @pytest.mark.parametrize("compound_coeff", [0, 1, 2])
    def test_from_predefined(self, model_type, compound_coeff, data, num_classes):
        model = model_type.from_predefined(compound_coeff, num_classes)
        assert isinstance(model, model_type)
        assert model.compound_coeff == compound_coeff
        del model

    def test_from_predefined_repeated_calls(self, model_type, data, num_classes):
        model0_1 = model_type.from_predefined(0, num_classes)
        model2_1 = model_type.from_predefined(2, num_classes)
        model2_2 = model_type.from_predefined(2, num_classes)
        model0_2 = model_type.from_predefined(0, num_classes)

        params0_1 = sum([x.numel() for x in model0_1.parameters()])
        params0_2 = sum([x.numel() for x in model0_2.parameters()])
        params2_1 = sum([x.numel() for x in model2_1.parameters()])
        params2_2 = sum([x.numel() for x in model2_2.parameters()])

        assert params2_1 == params2_2
        assert params0_1 == params0_2
        assert params0_2 < params2_1

        print(f"Params: {params2_1}")
        assert params2_1 > 5e6

    def test_create_boxes(self, model_type):
        num_classes = 2
        strides = [8, 16, 32, 64, 128]
        base_size = 512
        sizes = [(base_size // stride,) * 2 for stride in strides]

        pred_cls = [torch.rand(2, num_classes, *size, requires_grad=True) for size in sizes]
        pred_reg = [torch.rand(2, 4, *size, requires_grad=True).mul(512).round() for size in sizes]
        pred_centerness = [torch.rand(2, 1, *size, requires_grad=True) for size in sizes]

        boxes, locations = model_type.create_boxes(pred_cls, pred_reg, pred_centerness, strides)

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

        final_pred, _ = model_type.create_boxes(cls_targets, reg_targets, centerness_targets, strides, threshold=0.5)
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
        center_radius = 2
        strides = [8, 16, 32, 64, 128]
        sizes = [(image_size // stride,) * 2 for stride in strides]

        img = torch.zeros(batch_size, 1, image_size, image_size)
        pred_cls = [torch.rand(batch_size, num_classes, *size).sub_(0.45).clamp_min_(0) for size in sizes]
        pred_cls[0].clamp_min(0.111)
        pred_reg = [torch.rand(batch_size, 4, *size).mul_(image_size / 4).round_().clamp_min_(24) for size in sizes]
        pred_centerness = [torch.rand(batch_size, 1, *size) for size in sizes]

        final_pred, _ = model_type.create_boxes(
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

        final_pred, _ = model_type.create_boxes(
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
        center_radius = 2
        strides = [8, 16, 32, 64, 128]
        sizes = [(image_size // stride,) * 2 for stride in strides]

        img = torch.zeros(batch_size, 1, image_size, image_size)
        pred_cls = [torch.rand(batch_size, num_classes, *size).sub_(0.45).clamp_min_(0) for size in sizes]
        pred_reg = [torch.rand(batch_size, 4, *size).mul_(image_size / 4).round_().clamp_min_(24) for size in sizes]
        pred_centerness = [torch.rand(batch_size, 1, *size) for size in sizes]

        batched, _ = model_type.create_boxes(
            pred_cls, pred_reg, pred_centerness, strides, threshold=0.5, nms_threshold=0.3
        )

        pred_cls = [x[0:1] for x in pred_cls]
        pred_reg = [x[0:1] for x in pred_reg]
        pred_centerness = [x[0:1] for x in pred_centerness]
        unbatched, _ = model_type.create_boxes(
            pred_cls, pred_reg, pred_centerness, strides, threshold=0.5, nms_threshold=0.3
        )

        unbatched.squeeze_(0)
        batched = batched[0]
        batched = batched[(batched != -1).all(dim=-1)]
        assert batched.shape == unbatched.shape
        assert torch.allclose(batched, unbatched)
