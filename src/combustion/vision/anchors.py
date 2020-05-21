#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


class AnchorBoxTransform(nn.Module):
    def __init__(self, mean: Optional[Tensor] = None, std: Optional[Tensor] = None, log_length: bool = False):
        super(AnchorBoxTransform, self).__init__()
        self.mean = mean
        self.std = std
        self.log_length = log_length

    def forward(self, boxes: Tensor, deltas: Tensor) -> Tensor:
        # calculate input boxes width/height/center
        widths = boxes[:, :, 2] - boxes[:, :, 0]
        heights = boxes[:, :, 3] - boxes[:, :, 1]
        center_x = boxes[:, :, 0] + 0.5 * widths
        center_y = boxes[:, :, 1] + 0.5 * heights

        # unapply mean/variance normalization on deltas
        if self.std is not None:
            deltas = deltas.mul(self.std)
        if self.mean is not None:
            deltas = deltas.add(self.mean)
        dx, dy, dw, dh = deltas.split(1, dim=-1)

        # unapply log on dh, dw
        if self.log_length:
            dw, dh = [torch.exp(x) for x in (dw, dh)]

        pred_center_x = center_x + dx * widths
        pred_center_y = center_y + dy * heights

        pred_w = dw * widths
        pred_h = dh * heights

        pred_boxes_x1 = pred_center_x - 0.5 * pred_w
        pred_boxes_y1 = pred_center_y - 0.5 * pred_h
        pred_boxes_x2 = pred_center_x + 0.5 * pred_w
        pred_boxes_y2 = pred_center_y + 0.5 * pred_h

        pred_boxes = torch.stack([pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2], dim=-1)
        return pred_boxes


class ClipBoxes(nn.Module):
    def __init__(self):
        super(ClipBoxes, self).__init__()

    def forward(self, boxes: Tensor, img: Tensor) -> Tensor:
        batch_size, num_channels, height, width = img.shape
        boxes[:, :, :2] = torch.clamp(boxes[:, :, :2], min=0)
        boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], max=width)
        boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], max=height)
        return boxes


class Anchors(nn.Module):
    def __init__(
        self,
        levels: List[int],
        strides: Optional[List[int]] = None,
        sizes: Optional[List[int]] = None,
        ratios: Optional[List[int]] = None,
        scales: Optional[List[int]] = None,
    ):
        super(Anchors, self).__init__()
        self.levels = levels

        if strides is None:
            self.strides = [2 ** x for x in self.levels]
        if sizes is None:
            self.sizes = [2 ** (x + 2) for x in self.levels]
        if ratios is None:
            self.ratios = torch.tensor([0.5, 1, 2])
        if scales is None:
            self.scales = torch.tensor([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

    def forward(self, image: Tensor) -> Tensor:
        image_shape = image.shape[-2:]
        image_shape = np.array(image_shape)
        image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in self.levels]

        all_anchors = []

        for idx, p in enumerate(self.levels):
            anchors = self._generate_anchors(base_size=self.sizes[idx], ratios=self.ratios, scales=self.scales)
            anchors = anchors.type_as(image)
            shifted_anchors = self._shift(image_shapes[idx], self.strides[idx], anchors)
            all_anchors.append(shifted_anchors)

        all_anchors = torch.cat(all_anchors, 0)
        return all_anchors

    def _generate_anchors(self, base_size: int, ratios: List[float], scales: List[float]) -> Tensor:
        num_anchors = len(ratios) * len(scales)
        anchors = torch.zeros(num_anchors, 4)
        anchors[:, 2:] = base_size * scales.repeat(2, len(ratios)).T

        areas = anchors[:, 2] * anchors[:, 3]
        anchors[:, 2] = (areas / ratios.repeat(len(scales))).sqrt()
        anchors[:, 3] = anchors[:, 2] * ratios.repeat(len(scales))
        anchors[:, 0::2] -= (anchors[:, 2] * 0.5).repeat(2, 1).T
        anchors[:, 1::2] -= (anchors[:, 3] * 0.5).repeat(2, 1).T
        return anchors

    def _shift(self, shape: Tensor, stride: Tensor, anchors: Tensor) -> Tensor:
        shift_x = torch.arange(0, shape[1]).type_as(anchors).add_(0.5).mul_(stride)
        shift_y = torch.arange(0, shape[0]).type_as(anchors).add_(0.5).mul_(stride)
        shift_y, shift_x = torch.meshgrid(shift_x, shift_y)
        shifts = torch.stack([t.reshape(-1) for t in (shift_x, shift_y) * 2], 0).T
        del shift_x
        del shift_y

        A = anchors.shape[0]
        K = shifts.shape[0]

        all_anchors = anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).permute(1, 0, 2)
        all_anchors = all_anchors.reshape((K * A, 4))
        return all_anchors
