#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional, Tuple

import torch
from torch import Tensor


try:
    from kornia.feature import non_maxima_suppression2d
except ImportError:

    def non_maxima_suppression2d(*args, **kwargs):
        raise ImportError(
            "PointsToAnchors requires kornia. "
            "Please install combustion with 'vision' extras using "
            "pip install combustion [vision]"
        )


class AnchorsToPoints:
    r"""Transform that converts bounding boxes to CenterNet style labels
    as described in the paper `Objects as Points`_.

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
        self.num_classes = abs(int(num_classes))
        self.downsample = abs(int(downsample))
        self.iou_threshold = abs(float(iou_threshold))
        self.radius_div = abs(float(radius_div))

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

        valid_indices = classes[..., -1] >= 0
        bbox, classes = bbox[valid_indices], classes[valid_indices]

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

        size_target_x = (x2 - x1).abs_()
        size_target_y = (y2 - y1).abs_()

        # if user gives zero-area bounding box there will be nan results
        bad_indices = (size_target_x == 0) | (size_target_y == 0)
        if bad_indices.any():
            raise RuntimeError(f"Found zero area bounding boxes:\n{bbox[bad_indices]}")

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
        sigma = (x2 - x1).mul_(self.iou_threshold).add_(x1).sub_(x2).div_(self.radius_div).abs_().clamp_(min=1e-6)
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
        mesh_y, mesh_x = torch.meshgrid(torch.arange(out_height), torch.arange(out_width))
        mesh_x = mesh_x.expand(num_rois, -1, -1).type_as(center_x)
        mesh_y = mesh_y.expand(num_rois, -1, -1).type_as(center_y)

        # gaussian splat following formula
        # Y = exp(-[(y - y_cent)**2 + (x - x_cent)**2] / [2 sigma ** 2])
        square_diff_x = (mesh_x - center_x.view(num_rois, 1, 1)).pow_(2)
        square_diff_y = (mesh_y - center_y.view(num_rois, 1, 1)).pow_(2)
        divisor = sigma.pow(2).mul_(2).view(num_rois, 1, 1).expand_as(square_diff_x)

        assert (divisor != 0).all(), "about to divide by zero, probably a zero area bbox"
        maps = (square_diff_x + square_diff_y).div_(divisor).neg_().exp()
        return maps


class PointsToAnchors:
    r"""Transform that converts CenterNet style labels to anchor boxes and class labels
    (i.e. reverses the transform performed by `AnchorsToPoints`) as described in the
    paper `Objects as Points`_. Anchor boxes are identified in the input as points
    that are greater than their 8 neighbors. The maximum number of boxes returned is
    parameterized, and selection is performed based on classification score. A threshold
    is can also be set such that scores below this threshold will not contribute to the
    output.

    Args:
        upsample (int):
            An integer factor by which the points will be upsampled to produce box coordinates.

        max_roi (int):
            The maximum number of boxes to include in the final output. Only the top `max_roi` scoring
            points will be converted into anchor boxes.

        threshold (float, optional):
            If given, discard boxes with classification scores less than or equal to `threshold`.
            Default 0.0

    Shape:
        - Points: :math:`(*, C + 4, H, W)` where :math:`C` is the number of classes, and :math:`H, W`
          are the height and width of the heatmap.
        - Output: :math:`(*, N, 6)` where :math:`*` means an optional batch dimension
          and :math:`N` is the number of output anchor boxes. Indices `0-3` of the output give
          the box coordinates :math:`(x1, y1, x2, y2)`, index `4` gives classification score,
          and index `5` gives the class label.

    .. _Objects as Points:
        https://arxiv.org/abs/1904.07850
    """

    def __init__(
        self, upsample: int, max_roi: int, threshold: float = 0.0,
    ):
        self.max_roi = int(max_roi)
        self.upsample = int(upsample)
        self.threshold = float(threshold)

    def __repr__(self):
        s = f"PointsToAnchors(upsample={self.upsample}"
        s += f", max_roi={self.max_roi}"
        if self.threshold > 0:
            s += f", threshold={self.threshold}"
        s += ")"
        return s

    def __call__(self, points: Tensor) -> Tensor:
        # batched recursion
        if points.ndim > 3:
            return self._batched_recurse(points)

        classes, regressions = points[:-4, :, :], points[-4:, :, :]
        height, width = classes.shape[-2:]
        classes.shape[-3]

        # identify maxima as points greater than their 8 neighbors
        classes = non_maxima_suppression2d(classes.unsqueeze(0), kernel_size=(3,) * 2).squeeze(0)

        # extract class / center x / center y indices of top k scores over heatmap
        topk = min(self.max_roi, classes.numel())
        nms_scores, nms_idx = classes.view(-1).topk(topk, dim=-1)
        nms_idx = nms_idx[nms_scores > self.threshold]
        nms_scores = nms_scores[nms_scores > self.threshold]

        # % / width
        center_x = (nms_idx % (height * width) % width).unsqueeze(-1)
        center_y = (nms_idx % (height * width) // width).unsqueeze(-1)
        cls = (nms_idx // (height * width)).unsqueeze(-1)

        offset_x = regressions[0, center_y, center_x]
        offset_y = regressions[1, center_y, center_x]
        size_x = regressions[2, center_y, center_x]
        size_y = regressions[3, center_y, center_x]

        # get upsampled centers by scaling up and applying offset
        center_x = center_x.float().mul_(self.upsample).add_(offset_x)
        center_y = center_y.float().mul_(self.upsample).add_(offset_y)

        # get box coordinates by applying height/width deltas about upsampled centers
        x1 = center_x - size_x.div(2)
        x2 = center_x + size_x.div(2)
        y1 = center_y - size_y.div(2)
        y2 = center_y + size_y.div(2)
        assert (x1 <= x2).all()
        assert (y1 <= y2).all()

        output = torch.cat([x1, y1, x2, y2, nms_scores.unsqueeze(-1), cls.float()], dim=-1)
        output = output[nms_scores > self.threshold]
        return output

    def _batched_recurse(self, points: Tensor) -> Tensor:
        assert points.ndim > 3
        batch_size = points.shape[0]

        # recurse on examples in batch
        results = []
        for elem in points:
            results.append(self(elem))

        # determine maximum number of boxes in example for padding
        max_roi = max([t.shape[0] for t in results])

        # combine examples into output batch
        output = torch.empty(batch_size, max_roi, 6).fill_(-1).type_as(points)
        for i, result in enumerate(results):
            num_roi = result.shape[0]
            output[i, :num_roi, ...] = result
        return output
