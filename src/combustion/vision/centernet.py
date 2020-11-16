#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Iterable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import ByteTensor, Tensor

from combustion.util import alpha_blend, apply_colormap, check_is_tensor, check_ndim_match

from .convert import to_8bit


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
        self,
        num_classes: int,
        downsample: int,
        iou_threshold: Optional[float] = 0.7,
        radius_div: Optional[float] = 3,
        min_sigma: float = 1e-6,
    ):
        self.num_classes = abs(int(num_classes))
        self.downsample = abs(int(downsample))
        self.iou_threshold = abs(float(iou_threshold))
        self.radius_div = abs(float(radius_div))
        self.min_sigma = abs(float(min_sigma))

    def __repr__(self):
        s = f"AnchorsToPoints(num_classes={self.num_classes}"
        s += f", R={self.downsample}"
        s += f", iou={self.iou_threshold}"
        s += f", radius_div={self.radius_div}"
        s += f", min_sigma={self.min_sigma}"
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
        sigma = (
            (x2 - x1).mul_(self.iou_threshold).add_(x1).sub_(x2).div_(self.radius_div).abs_().clamp_min_(self.min_sigma)
        )
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

        max_roi (int or ``None``):
            The maximum number of boxes to include in the final output. Only the top `max_roi` scoring
            points will be converted into anchor boxes. If ``None``, don't drop boxes below a score threshold.

        threshold (float, optional):
            If given, discard boxes with classification scores less than or equal to `threshold`.
            Default 0.0

        return_indices (bool):
            If ``True`` is passed to :func:`forward`, also return indices of the points that were mapped to boxes

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
        self,
        upsample: int,
        max_roi: Optional[int] = None,
        threshold: float = 0.0,
    ):
        self.max_roi = int(max_roi) if max_roi is not None else None
        self.upsample = int(upsample)
        self.threshold = float(threshold)

    def __repr__(self):
        s = f"PointsToAnchors(upsample={self.upsample}"
        s += f", max_roi={self.max_roi}"
        if self.threshold > 0:
            s += f", threshold={self.threshold}"
        s += ")"
        return s

    def __call__(self, points: Tensor, return_indices=False) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        # batched recursion
        if points.shape[-3] <= 4:
            raise ValueError(f"Expected points.shape[-3] > 4, found shape {points.shape}")
        if points.ndim == 4:
            return self._batched_recurse(points, return_indices)
        elif points.ndim > 4:
            raise ValueError(f"Expected points.ndim <=, found shape {points.ndim}")

        classes, regressions = points[:-4, :, :], points[-4:, :, :]
        height, width = classes.shape[-2:]
        classes.shape[-3]

        # identify maxima as points greater than their 8 neighbors
        classes = non_maxima_suppression2d(classes.unsqueeze(0), kernel_size=(3,) * 2).squeeze(0)

        # extract class / center x / center y indices of top k scores over heatmap
        if self.max_roi is not None:
            topk = min(self.max_roi, classes.numel())
            nms_scores, nms_idx = classes.view(-1).topk(topk, dim=-1)
        else:
            nms_scores, nms_idx = classes.view(-1).sort(dim=-1, descending=True)

        nms_idx = nms_idx[nms_scores > self.threshold]
        nms_scores = nms_scores[nms_scores > self.threshold]

        # build a mask of indices that were mapped to boxes
        if return_indices:
            final_indices = torch.zeros_like(classes.view(-1), dtype=torch.bool)
            final_indices[nms_idx] = True
            final_indices = final_indices.view_as(classes).nonzero(as_tuple=False)

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
        # output = output[nms_scores > self.threshold]
        if return_indices:
            return output, final_indices
        else:
            return output

    def _batched_recurse(self, points: Tensor, return_indices: bool) -> Tensor:
        assert points.ndim > 3
        batch_size = points.shape[0]

        # recurse on examples in batch
        results = []
        for elem in points:
            results.append(self(elem, return_indices))

        # split apart the result
        if return_indices:
            boxes = [x[0] for x in results]
            indices = [x[1] for x in results]
        else:
            boxes = results
            indices = None

        # determine maximum number of boxes in example for padding
        max_roi = max([t.shape[0] for t in boxes])

        # combine examples into output batch
        output = torch.empty(batch_size, max_roi, 6).fill_(-1).type_as(points)
        for i, result in enumerate(boxes):
            num_roi = result.shape[-2]
            output[i, :num_roi, :] = result

        if return_indices:
            output_indices = torch.empty(batch_size, max_roi, 4, device=output.device).fill_(-1).long()
            for i, (src, dest) in enumerate(zip(indices, output_indices)):
                # add batch index
                batch_idx = torch.tensor([[i]]).type_as(src).expand(src.shape[-2], -1)
                src = torch.cat([batch_idx, src], dim=-1)
                dest[: src.shape[-2], :] = src.type_as(dest)

        if return_indices:
            return output, output_indices
        return output


class CenterNetMixin:
    PAD_VALUE: float = -1

    @staticmethod
    def split_point_target(target: Tensor) -> Tuple[Tensor, Tensor]:
        r"""Split a CenterNet target into heatmap and regression components.

        .. note::
            This operation returns views of the original tensor.

        Args:
            target (:class:`torch.Tensor`):
                The target to split.

        Shape:
            * ``target`` - :math:`(*, C + 4, H, W)` where :math:`C` is the number of classes.
            * Output - :math:`(*, C, H, W)` and :math:`(*, 4, H, W)`
        """
        check_is_tensor(target, "target")
        heatmap = target[..., :-4, :, :]
        bbox = target[..., -4:, :, :]
        return heatmap, bbox

    @staticmethod
    def split_regression(regression: Tensor) -> Tuple[Tensor, Tensor]:
        r"""Split a CenterNet regression prediction into offset and sizecomponents.

        .. note::
            This operation returns views of the original tensor.

        Args:
            regression (:class:`torch.Tensor`):
                The target to split.

        Returns:
            Tuple of offset and size tensors

        Shape:
            * ``target`` - :math:`(*, 4, H, W)`
            * Output - :math:`(*, 2, H, W)` and :math:`(*, 2, H, W)`
        """
        check_is_tensor(regression, "regression")
        offset = regression[..., :2, :, :]
        size = regression[..., 2:, :, :]
        assert offset.shape[-3] == 2
        assert size.shape[-3] == 2
        return offset, size

    @staticmethod
    def combine_point_target(heatmap: Tensor, regression: Tensor) -> Tensor:
        r"""Combine a CenterNet heatmap and regression components into a single label.

        Args:
            heatmap (:class:`torch.Tensor`):
                The CenterNet heatmap.

            regression (:class:`torch.Tensor`):
                The CenterNet regression map.

        Shape:
            * ``heatmap`` - :math:`(*, C, H, W)`
            * ``regression`` - :math:`(*, 4, H, W)`
            * Output - :math:`(*, C+4, H, W)`
        """
        # input validation
        check_is_tensor(heatmap, "heatmap")
        check_is_tensor(regression, "regression")
        if regression.shape[-3] != 4:
            raise ValueError(f"Expected regression.shape[-3] == 4, found shape {regression.shape}")

        return torch.cat([heatmap, regression], dim=-3)

    @staticmethod
    def combine_bbox_scores_class(bbox: Tensor, cls: Tensor, scores: Tensor, *extra_scores) -> Tensor:
        r"""Combine a bounding box coordinates and labels into a single label. Combined tensor
        will be ordered along the last dimension as ``bbox``, ``scores``, ``cls``.

        Args:
            bbox (:class:`torch.Tensor`):
                Coordinates of the bounding box.

            cls (:class:`torch.Tensor`):
                Class associated with each bounding box

            scores (:class:`torch.Tensor`):
                Probability associated with each bounding box.

            *extra_scores (:class:`torch.Tensor`):
                Additional scores to combine

        Shape:
            * ``bbox`` - :math:`(*, N, 4)`
            * ``scores`` - :math:`(*, N, S)`
            * ``cls`` - :math:`(*, N, 1)`
            * Output - :math:`(*, N, 4 + S + 1)`
        """
        # input validation
        check_is_tensor(bbox, "bbox")
        check_is_tensor(scores, "scores")
        check_is_tensor(cls, "cls")
        if bbox.shape[-1] != 4:
            raise ValueError(f"Expected bbox.shape[-1] == 4, found shape {bbox.shape}")
        return torch.cat([bbox, scores, *extra_scores, cls], dim=-1)

    @staticmethod
    def combine_regression(offset: Tensor, size: Tensor) -> Tensor:
        r"""Combines CenterNet offset and size predictions into a single tensor.

        Args:
            offset (:class:`torch.Tensor`):
                Offset component of the heatmap

            size (:class:`torch.Tensor`):
                Size component of the heatmap

        Returns:
            Tuple of offset and size tensors

        Shape:
            * ``offset`` - :math:`(*, 2, H, W)`
            * ``size`` - :math:`(*, 2, H, W)`
            * Output - :math:`(*, 4, H, W)`
        """
        check_is_tensor(offset, "offset")
        check_is_tensor(size, "size")
        return torch.cat([offset, size], dim=-3)

    @staticmethod
    def heatmap_max_score(heatmap: Tensor) -> Tensor:
        r"""Computes global maximum scores over a heatmap on a per-class basis.

        Args:
            heatmap (:class:`torch.Tensor`):
                CenterNet heatmap

        Shape:
            * ``heatmap`` - :math:`(*, C+4, H, W)`
            * Output - :math:`(*, C)`
        """
        check_is_tensor(heatmap, "heatmap")
        heatmap = heatmap[..., :-4, :, :]
        non_spatial_shape = heatmap.shape[:-2]
        output = heatmap.view(*non_spatial_shape, -1).max(dim=-1).values
        return output

    @staticmethod
    def visualize_heatmap(
        heatmap: Tensor,
        background: Optional[Tensor] = None,
        cmap: str = "gnuplot",
        same_on_batch: bool = True,
        heatmap_alpha: float = 0.5,
        background_alpha: float = 0.5,
    ) -> List[ByteTensor]:
        r"""Generates visualizations of a CenterNet heatmap. Can optionally overlay the
        heatmap on top of a background image.

        Args:
            heatmap (:class:`torch.Tensor`):
                The heatmap to visualize

            background (:class:`torch.Tensor`):
                An optional background image for the heatmap visualization

            cmap (str):
                Matplotlib colormap

            same_on_batch (bool):
                See :func:`combustion.vision.to_8bit`

            heatmap_alpha (float):
                See :func:`combustion.util.alpha_blend`

            background_alpha (float):
                See :func:`combustion.util.alpha_blend`

        Returns:
            List of tensors, where each tensor is a heatmap visualization for one class in the heatmap

        Shape:
            * ``heatmap`` - :math:`(N, C, H, W)` where :math:`C` is the number of classes in the heatmap.
            * Output - :math:`(N, 3, H, W)`
        """
        check_is_tensor(heatmap, "heatmap")
        if background is not None:
            check_is_tensor(background, "heatmap")
            # need background to be float [0, 1] for alpha blend w/ heatmap
            background = to_8bit(background, same_on_batch=same_on_batch).float().div_(255).cpu()

            if background.shape[-3] == 1:
                repetitions = [
                    1,
                ] * background.ndim
                repetitions[-3] = 3
                background = background.repeat(*repetitions)

        num_channels = heatmap.shape[-3]

        result = []
        for channel_idx in range(num_channels):
            _ = heatmap[..., channel_idx : channel_idx + 1, :, :]
            _ = to_8bit(_, same_on_batch=same_on_batch)

            # output is float from [0, 1]
            heatmap_channel = apply_colormap(_.cpu(), cmap=cmap)

            # drop alpha channel
            heatmap_channel = heatmap_channel[..., :3, :, :]

            # alpha blend w/ background
            if background is not None:
                heatmap_channel = F.interpolate(
                    heatmap_channel, size=background.shape[-2:], mode="bilinear", align_corners=True
                )
                heatmap_channel = alpha_blend(heatmap_channel, background, heatmap_alpha, background_alpha)[0]

            heatmap_channel = heatmap_channel.mul_(255).byte()
            result.append(heatmap_channel)

        return result

    @staticmethod
    def append_heatmap_label(old_label: Tensor, new_label: Tensor) -> Tensor:
        r"""Adds a new label element to an existing CenterNet target.
        The new label will be concatenated to along the heatmap channel dimension
        immediately preceeding the regression component of the heatmap.

        Args:
            old_label (:class:`torch.Tensor`):
                The existing heatmap label

            new_label (:class:`torch.Tensor`):
                The heatmap channel to add to ``old_label``

        Shape:
            * ``old_label`` - :math:`(*, C_0 + 4, H, W)`
            * ``new_label`` - :math:`(*, C_1, H, W)`
            * Output - :math:`(*, C_0 + C_1 + 4, H, W)`
        """
        check_is_tensor(old_label, "old_label")
        check_is_tensor(new_label, "new_label")
        check_ndim_match(old_label, new_label, "old_label", "new_label")
        return torch.cat([old_label[..., :-4, :, :], new_label, old_label[..., -4:, :, :]], dim=-3)

    @staticmethod
    def get_global_pred_target_pairs(pred: Tensor, target: Tensor, pad_value: float = -1) -> Tensor:
        r"""Given predicted CenterNet heatmap and target bounding box label, create a paring of
        per-class global heatmap maxima to binary labels indicating whether or not the class was
        present in the true label.

        Args:
            pred (:class:`torch.Tensor`):
                Predicted heatmap.

            target (:class:`torch.Tensor`):
                Target bounding boxes in format ``x1, y1, x2, y2, class``.

            pad_value (float):
                Value used for padding a batched ``target`` input.

        Returns:
            Tensor paring a predicted probability with a binary indicator

        Shape:
            * ``pred`` - :math:`(*, C+4, H, W)`
            * ``target`` - :math:`(*, N_i, 5)`
            * Output - :math:`(*, C, 2)`
        """
        check_is_tensor(pred, "pred")
        check_is_tensor(target, "target")
        is_batch = pred.ndim > 3
        assert pred.shape[-3] > 4

        if is_batch:
            assert target.ndim > 2, "pred batched but target not batched"
            batch_result = []
            for pred_i, target_i in zip(pred, target):
                result = CenterNetMixin.get_global_pred_target_pairs(pred_i, target_i, pad_value)
                batch_result.append(result)
            return torch.stack(batch_result, dim=0)

        # we might be operating on a batched example that was padded, so remove these padded locations
        pad_locations = (target == -1).all(dim=-1)
        target = target[~pad_locations, :]

        # get the global max probability for each class in the heatmap
        num_classes = pred.shape[-3] - 4
        max_pred_scores = CenterNetMixin.heatmap_max_score(pred)

        # get boolean mask of which classes were present in the target
        target_class_present = torch.zeros(num_classes, device=target.device).bool()
        target_class_present[target[..., -1].unique().long()] = True

        assert max_pred_scores.shape == target_class_present.shape
        return torch.stack([max_pred_scores, target_class_present.type_as(max_pred_scores)], dim=-1)

    @staticmethod
    def filter_heatmap_classes(
        heatmap: Tensor, keep_classes: Iterable[int], return_inverse: bool = False, with_regression: bool = False
    ) -> Tensor:
        r"""Filters a CenterNet heatmap based on class, dropping class channels that do not meet the critera.

        Args:
            heatmap (:class:`torch.Tensor`):
                Heatmap to filter

            keep_classes (iterable of ints):
                Integer id of the classes to keep

            return_inverse (:class:`torch.Tensor`):
                If ``True``, remove channels with classes not in ``keep_classes``

            with_regression (bool):
                If ``True``, expect :math:`C+4` channels in ``heatmap``

        Shape:
            * ``target`` - :math:`(*, C, H, W)` or :math:`(*, C+4, H, W)`, where :math:`C` is the number of classes
            * Output - :math:`(*, C', H, W)` or :math:`(*, C'+4, H, W)`
        """
        check_is_tensor(heatmap, "heatmap")
        if not isinstance(keep_classes, Iterable):
            raise TypeError(f"Expected iterable for keep_classes, found {type(keep_classes)}")
        if not keep_classes:
            raise ValueError(f"Expected non-empty iterable for keep classes, found {keep_classes}")

        if with_regression:
            num_classes = heatmap.shape[-3] - 4
        else:
            num_classes = heatmap.shape[-3]
        assert num_classes > 0

        possible_classes = set(range(num_classes))
        keep_classes = set(keep_classes) if not return_inverse else possible_classes - set(keep_classes)

        keep_heatmap = heatmap[..., tuple(keep_classes), :, :]

        if with_regression:
            return torch.cat([keep_heatmap, heatmap[..., -4:, :, :]], dim=-3)
        else:
            return keep_heatmap.clone()
