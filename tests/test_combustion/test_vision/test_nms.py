#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch

from combustion.vision import nms


def test_nms_batched():
    boxes = torch.tensor([[0, 0, 10, 10], [1, 1, 11, 11], [10, 10, 20, 20],]).unsqueeze(0).float()

    scores = torch.tensor([0.1, 0.5, 0.05,]).unsqueeze(0)

    indices = nms(boxes, scores, 0.5)

    batch_expected = torch.tensor([0, 0])
    box_expected = torch.tensor([1, 2])

    assert indices[0].shape == batch_expected.shape
    assert indices[1].shape == box_expected.shape

    assert torch.allclose(indices[0], batch_expected)
    assert torch.allclose(indices[1], box_expected)

    nms_boxes, nms_scores = boxes[indices], scores[indices]
    assert torch.allclose(nms_boxes, boxes[0, (1, 2), :])
    assert torch.allclose(nms_scores, scores[0, (1, 2)])


def test_nms_unbatched():
    boxes = torch.tensor([[0, 0, 10, 10], [1, 1, 11, 11], [10, 10, 20, 20],]).float()

    scores = torch.tensor([0.1, 0.5, 0.05,])

    indices = nms(boxes, scores, 0.5)

    assert torch.allclose(indices, torch.tensor([1, 2]))

    nms_boxes, nms_scores = boxes[indices], scores[indices]
    assert torch.allclose(nms_boxes, boxes[(1, 2), :])
    assert torch.allclose(nms_scores, scores[(1, 2),])
