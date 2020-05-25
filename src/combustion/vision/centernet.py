#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional, Tuple

import torch
from torch import Tensor


class AnchorsToPoints:
    r"""Transform that converts bounding boxes to CenterNet style labels
    as described in the paper `Objects as Points:`_ .

    Transformed outputs are as follows
        * probabilities of a pixel being a box center of class `i` with gaussian smoothing
        * x and y coordinates of the bounding box center in a downsampled grid
        * height and width of the anchor box

    Args:
        num_classes (int):
            The number of possible classes for classification.

        downsample (int):
            An integer factor by which the image size will be downsampled to produce
            the label heatmap.

        iou_threshold (float, optional)
            The IoU threshold required for a possible anchor box to be considered
            in the calculation of the Gaussian smoothing sigma. Default 0.7.

        radius_div (float, optional):
            The factor by which the radius of all possible anchor boxes with IoU > threshold
            will be divided to determine the Gaussian smoothing sigma. Default 3.

    Shape:
        - Bounding boxes: :math:`(*, N, 4)` where :math:`*` means an optional batch dimension
          and :math:`N` is the number of bounding boxes
        - Classes: :math:`(*, N, 1)`
        - Output: :math:`(*, C + 4, H, W)` where :math:`C` is the number of classes, and :math:`H, W`
          are the height and width of the downsampled heatmap.

    .. _Objects as Points:
        https://arxiv.org/abs/1904.07850
    """

    def __init__(
        self, num_classes: int, downsample: int, iou_threshold: Optional[float] = 0.7, radius_div: Optional[float] = 3
    ):
        self.num_classes = int(num_classes)
        self.downsample = int(downsample)
        self.iou_threshold = float(iou_threshold)
        self.radius_div = float(radius_div)

    def __repr__(self):
        s = f"AnchorsToPoints(num_classes={self.num_classes}"
        s += f", R={self.downsample}"
        s += f", iou={self.iou_threshold}"
        s += f", radius_div={self.radius_div}"
        s += ")"
        return s

    def __call__(self, bbox: Tensor, classes: Tensor, shape: Tuple[int, int]) -> Tensor:
        original_ndim = bbox.ndim

        # recurse on batched input
        if original_ndim == 3:
            results = []
            for box, cls in zip(bbox, classes):
                results.append(self(box, cls, shape))
            return torch.stack(results, 0)

        # unsqueeze a batch dim if not present
        bbox = bbox.view(bbox.shape[-2], 4)
        classes = classes.view(classes.shape[-2], 1)

        # determine input/output height/width
        height, width = shape[-2:]
        num_rois = bbox.shape[-2]
        out_height = height // self.downsample
        out_width = width // self.downsample

        # regression targets for true box size
        x1, y1 = bbox[..., 0], bbox[..., 1]
        x2, y2 = bbox[..., 2], bbox[..., 3]

        size_target_x = x2 - x1
        size_target_y = y2 - y1

        # center x/y coords
        center_x = (x2 + x1).div_(2)
        center_y = (y2 + y1).div_(2)

        # all other steps are performed in the downsampled space p/R
        # for t in (x1, y1, x2, y2):
        # t.floor_divide_(self.downsample)
        x1 = x1.floor_divide(self.downsample)
        y1 = y1.floor_divide(self.downsample)
        x2 = x2.floor_divide(self.downsample)
        y2 = y2.floor_divide(self.downsample)

        # local offsets of centers in downsampled space
        # this is used to recover discretization error of centers from downsample
        offset_target_x = center_x - (x2 + x1).floor_divide_(2).mul_(self.downsample)
        offset_target_y = center_y - (y2 + y1).floor_divide_(2).mul_(self.downsample)

        # center x/y coords in downsampled space
        center_x = center_x.floor_divide(self.downsample)
        center_y = center_y.floor_divide(self.downsample)

        # assign to reg targets tensor
        reg_targets = torch.empty(4, out_height, out_width).type_as(bbox).fill_(-1).float()
        y_ind = center_y.long()
        x_ind = center_x.long()
        reg_targets[0, y_ind, x_ind] = offset_target_x
        reg_targets[1, y_ind, x_ind] = offset_target_y
        reg_targets[2, y_ind, x_ind] = size_target_x
        reg_targets[3, y_ind, x_ind] = size_target_y

        # the next step is to splat downsampled true centers onto a heatmap using a gaussian dist.
        # the gaussian sigma is determined as follows:
        #   1.  find all possible box corner positions such that a minimum IoU
        #       is maintained w.r.t original box
        #   2.  determine the radius from the original corner that encloses all such new positions
        #   3.  calculate sigma as this radius divided by some scalar
        #
        # we want to find a larger rectangle w/ IoU >= threshold w.r.t original box
        # corners of this rectangle (preserving aspect ratio) are x1, y1, (x2+r), y2
        # so we compute area of larger rectangle and solve for r
        #   threshold = A / A'
        #   threshold = [(x2 + r - x1)(y2 - y1)] / [(x2 - x1)(y2 - y1)]
        #   threshold = (x2 + r - x1) / (x2 - x1)
        # so...
        #   r = threshold * (x2 - x1) + x1 - x2
        #   sigma = r / c
        #   => sigma = [threshold * (x2 - x1) + x1 - x2] / c
        sigma = (x2 - x1).mul_(self.iou_threshold).add_(x1).sub_(x2).div_(self.radius_div).abs_()
        kernel_size = (sigma * 3).round_().long()
        kernel_size = torch.where(kernel_size * 2 != 0, kernel_size, kernel_size + 1)
        heatmap = self._gaussian_splat(num_rois, center_x, center_y, sigma, out_height, out_width)

        # combine heatmaps of same classes within a batch using element-wise maximum
        # TODO this isn't clean, can it be done without looping?
        cls_targets = torch.zeros(self.num_classes, out_height, out_width).type_as(bbox).float()
        for i in range(self.num_classes):
            if (classes == i).any():
                class_heatmap = heatmap[classes.view(num_rois) == i, ...].view(-1, out_height, out_width)
                cls_targets[i, ...] = class_heatmap.max(dim=0).values

        output = torch.cat([cls_targets, reg_targets], 0)
        return output

    def _gaussian_splat(self, num_rois, center_x, center_y, sigma, out_height, out_width) -> Tensor:
        mesh_x, mesh_y = torch.meshgrid(torch.arange(out_height), torch.arange(out_width))
        mesh_x = mesh_x.expand(num_rois, -1, -1).type_as(center_x)
        mesh_y = mesh_y.expand(num_rois, -1, -1).type_as(center_y)

        # gaussian splat following formula
        # Y = exp(-[(y - y_cent)**2 + (x - x_cent)**2] / [2 sigma ** 2])
        square_diff_x = (mesh_x - center_x.view(num_rois, 1, 1)).pow_(2)
        square_diff_y = (mesh_y - center_y.view(num_rois, 1, 1)).pow_(2)
        divisor = sigma.pow(2).mul_(2).view(num_rois, 1, 1).expand_as(square_diff_x)
        maps = (square_diff_x + square_diff_y).div_(divisor).neg_().exp()
        return maps
