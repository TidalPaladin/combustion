#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import torch
from combustion.nn.modules.transformer.detr import DETRHead, DETRCriterion, HungarianMatcher, DETRPrediction, DETRTarget

class TestDETRHead:

    def test_forward(self):
        B, D, H, W = 2, 32, 128, 128
        fpn = [torch.rand(B, D, H // 2 ** i, W // 2 ** i) for i in range(3)]
        num_boxes = 20
        num_layers = 2
        num_classes = 3
        layer = DETRHead(512, D, num_boxes, num_layers, num_classes)
        out = layer(fpn)

class TestDETRCriterion:

    def test_matching(self):
        coords = torch.tensor([
            [0.5, 0.5, 0.05, 0.05],
            [0.25, 0.25, 0.1, 0.1]
        ]).unsqueeze(0)
        classes = torch.tensor([
            [0],
            [0],
        ]).unsqueeze(0)

        pred_coords1 = torch.tensor([[
            [0.5, 0.5, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.1],
        ]])
        pred_coords2 = torch.tensor([[
            [0.1, 0.1, 0.1, 0.1],
            [0.5, 0.5, 0.1, 0.1],
        ]])
        logits = torch.tensor([[
            [10, -10, -10],
            [10, -10, -10],
        ]])

        matcher = HungarianMatcher()
        match1 = matcher(logits, pred_coords1, classes, coords)
        match2 = matcher(logits, pred_coords2, classes, coords)

        cost1 = (pred_coords1.view(-1, 1, 4) - coords.view(1, -1, 4)).abs().sum(dim=-1)
        cost2 = (pred_coords2.view(-1, 1, 4) - coords.view(1, -1, 4)).abs().sum(dim=-1)
        assert torch.allclose(cost1, cost2.roll(-2))

        t1 = torch.stack([match1.source_idx, match1.target_idx], dim=-1)
        t2 = torch.stack([match2.source_idx, match2.target_idx], dim=-1)
        assert (t1[..., 0] == t2[..., 0]).all()
        assert (t1[..., 1] == t2[..., 1].roll(-1)).all()

    def test_matching2(self):
        torch.random.manual_seed(42)
        coords = torch.rand(2, 32, 4)
        classes = torch.randint(0, 3, (2, 32, 1))

        pred_coords = torch.rand_like(coords)
        logits = torch.rand(2, 32, 3).logit()

        matcher = HungarianMatcher()
        match1 = matcher(logits, pred_coords, classes, coords)
        applied1 = coords[match1.batch_idx, match1.target_idx]

        rolled_coords = coords.roll(1, dims=-2)
        rolled_classes = classes.roll(1, dims=-2)
        match2 = matcher(logits, pred_coords, rolled_classes, rolled_coords)
        applied2 = rolled_coords[match2.batch_idx, match2.target_idx]

        assert torch.allclose(applied1, applied2)

    def test_matching3(self):
        torch.random.manual_seed(42)
        coords = torch.rand(2, 32, 4)
        classes = torch.randint(0, 1, (2, 32, 1))

        pred_coords = coords.clone()
        logits = torch.empty(2, 32, 2)
        logits[..., 0] = 10
        logits[..., -1] = -10

        matcher = HungarianMatcher()
        match = matcher(logits, pred_coords, classes, coords)
        assert (match.source_idx == match.target_idx).all()

    def test_forward(self, cuda):
        B, D, H, W = 2, 32, 128, 128
        fpn = [torch.rand(B, D, H // 2 ** i, W // 2 ** i) for i in range(3)]
        num_boxes = 20
        num_layers = 2
        num_classes = 3

        true_cls = torch.randint(0, num_classes, (B, 10, 1))
        true_box = torch.rand(B, 10, 4) * 127
        true_box[..., -2:] = torch.max(true_box[..., -2:], true_box[..., :2]+1)

        layer = DETRHead(512, D, num_boxes, num_layers, num_classes)

        if cuda:
            layer = layer.cuda()
            fpn = [x.cuda() for x in fpn]
            true_cls = true_cls.cuda()
            true_box = true_box.cuda()

        out = layer(fpn)
        out = out.replace(logits=out.logits.half(), coords=out.coords.half())

        criterion = DETRCriterion()
        loss = criterion(out, true_cls, true_box, (H, W))
        loss.total_loss.backward()

    def test_forward_empty(self, cuda):
        B, D, H, W = 2, 32, 128, 128
        fpn = [torch.rand(B, D, H // 2 ** i, W // 2 ** i) for i in range(3)]
        num_boxes = 20
        num_layers = 2
        num_classes = 3

        true_cls = torch.randint(0, num_classes, (B, 0, 1))
        true_box = torch.rand(B, 0, 4) 

        layer = DETRHead(512, D, num_boxes, num_layers, num_classes)

        if cuda:
            layer = layer.cuda()
            fpn = [x.cuda() for x in fpn]
            true_cls = true_cls.cuda()
            true_box = true_box.cuda()

        out = layer(fpn)
        out = out.replace(logits=out.logits.half(), coords=out.coords.half())

        criterion = DETRCriterion()
        loss = criterion(out, true_cls, true_box, (H, W))
        loss.total_loss.backward()


    def test_zero_loss(self):
        torch.random.manual_seed(42)
        coords = torch.rand(2, 32, 4)
        classes = torch.randint(0, 1, (2, 32, 1))

        pred_coords = coords.clone()
        logits = torch.empty(2, 32, 2)
        logits[..., 0] = 10
        logits[..., -1] = -10

        x1y1 = coords[..., :2] - coords[..., 2:] / 2
        x2y2 = coords[..., :2] + coords[..., 2:] / 2
        coords2 = torch.cat([x1y1, x2y2], dim=-1)

        pred = DETRPrediction(logits, pred_coords)
        true = DETRTarget(classes, coords2)
        criterion = DETRCriterion()
        loss = criterion.compute_from_prepared(pred, true)
        assert loss.score_loss == 0


    def test_matching_padding(self):
        torch.random.manual_seed(42)
        coords = torch.rand(2, 32, 4)
        classes = torch.randint(0, 3, (2, 32, 1))
        classes[0, -10:, ...] = -1
        classes[1, -12:, ...] = -1
        coords[0].mul_(10)
        coords[0, -10:, ...] = -1
        coords[1, -12:, ...] = -1

        pred_coords = torch.rand_like(coords)
        logits = torch.rand(2, 32, 3).logit()

        matcher = HungarianMatcher()
        match1 = matcher(logits, pred_coords, classes, coords)
        applied1 = coords[match1.batch_idx, match1.target_idx]

        rolled_coords = coords.roll(1, dims=-2)
        rolled_classes = classes.roll(1, dims=-2)
        match2 = matcher(logits, pred_coords, rolled_classes, rolled_coords)
        applied2 = rolled_coords[match2.batch_idx, match2.target_idx]

        assert torch.allclose(applied1, applied2)
