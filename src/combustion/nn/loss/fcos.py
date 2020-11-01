#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import Tensor
import torch.nn as nn
from .ciou import CompleteIoULoss
from .focal import FocalLossWithLogits
from combustion.util import check_is_tensor, check_dimension, check_dimension_match, check_shape, check_shapes_match
from combustion.vision.centernet import CenterNetMixin
from typing import List, Tuple, Optional, Union

# FCOS uses each FPN level to predict targets of different sizes
# This is the size range used in the paper
DEFAULT_INTEREST_RANGE: Tuple[Tuple[int, int], ...] = (
    (-1, 64),   # stride=8
    (64, 128),  # stirde=16
    (128, 256), # stride=32
    (256, 512), # stride=64
    (512, 10000000), # stride=128
)

IGNORE = -1

class FCOSLoss:
    r"""Implements the loss function and target creation as described in PLACEHOLDER.
    """


    def __init__(
        self, 
        strides: Tuple[int, ...],
        size_targets: Tuple[Tuple[int, int], ...],
        num_classes: int,
        interest_range: Tuple[Tuple[int, int], ...] = DEFAULT_INTEREST_RANGE,
        gamma: float = 2.0, 
        alpha: float = 0.5, 
        radius: Optional[int] = 1,
        pad_value: float = -1
    ):
        self.strides = tuple([int(x) for x in strides])
        self.size_targets = tuple([(int(x), int(y)) for x, y in size_targets])
        self.interest_range = tuple([(int(x), int(y)) for x, y in interest_range])
        self.num_classes = int(num_classes)
        self.pad_value = pad_value

        self.pad_value = float(pad_value)
        self.radius = int(radius) if radius is not None else None

        self.cls_criterion = FocalLossWithLogits(gamma, alpha, reduction="none")
        self.reg_criterion = CompleteIoULoss(reduction="none")
        self.centerness_criterion = nn.BCEWithLogitsLoss(reduction="none")

    def __call__(
        self, 
        cls_pred: Tuple[Tensor, ...], 
        reg_pred: Tuple[Tensor, ...], 
        centerness_pred: Tuple[Tensor, ...], 
        target_bbox: Tensor, 
        target_cls: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        batch_size = target_bbox.shape[0]
        cls_loss, reg_loss, centerness_loss = [], [], []

        for i in range(batch_size):
            cls = [t[i] for t in cls_pred]
            reg = [t[i] for t in reg_pred]
            centerness = [t[i] for t in centerness_pred]
            _target_bbox = self._drop_padding(target_bbox[i])
            _target_cls = self._drop_padding(target_cls[i])
            _cls_loss, _reg_loss, _centerness_loss = self.compute_from_box_target(
                cls, 
                reg, 
                centerness, 
                _target_bbox, 
                _target_cls
            )
            cls_loss.append(_cls_loss)
            reg_loss.append(_reg_loss)
            centerness_loss.append(_centerness_loss)
        
        cls_loss = sum(cls_loss)
        reg_loss = sum(reg_loss)
        centerness_loss = sum(centerness_loss)
        return cls_loss, reg_loss, centerness_loss

    def _drop_padding(self, x: Tensor) -> Tensor:
        padding = (x == self.pad_value).all(dim=-1)
        return x[~padding]

    def compute_from_box_target(
        self, 
        cls_pred: Tuple[Tensor, ...], 
        reg_pred: Tuple[Tensor, ...], 
        centerness_pred: Tuple[Tensor, ...], 
        target_bbox: Tensor, 
        target_cls: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        fcos_targets = self.create_targets(target_bbox, target_cls)
        return self.compute_from_fcos_target(cls_pred, reg_pred, centerness_pred, fcos_targets)

    def compute_from_fcos_target(
        self, 
        cls_pred: Tuple[Tensor, ...], 
        reg_pred: Tuple[Tensor, ...], 
        centerness_pred: Tuple[Tensor, ...], 
        fcos_target: Tuple[Tuple[Tensor, Tensor, Tensor], ...], 
    ) -> Tuple[Tensor, Tensor, Tensor]:
        cls_loss = [
            self.cls_criterion(pred, true) 
            for pred, (true, _, _) in zip(cls_pred, fcos_target)
        ]
        centerness_loss = [
            self.centerness_criterion(pred, true) 
            for pred, (_, _, true) in zip(centerness_pred, fcos_target)
        ]

        reg_loss = [
            self.reg_criterion(pred.view(4, -1).permute(1, 0), true.view(4, -1).permute(1, 0)).view(pred.shape[1:]).unsqueeze_(0)
            for pred, (_, true, _) in zip(reg_pred, fcos_target)
        ]

        cls_loss = sum([x.sum() for x in cls_loss])
        reg_loss = sum([x.sum() for x in reg_loss])
        centerness_loss = sum([x.sum() for x in centerness_loss])

        return cls_loss, reg_loss, centerness_loss

    def create_targets(
        self,
        bbox: Tensor, 
        cls: Tensor, 
    ) -> Tuple[Tuple[Tensor, Tensor, Tensor], ...]:
        class_targets, reg_targets, centerness_targets = [], [], []
        for irange, stride, size_target in zip(self.interest_range, self.strides, self.size_targets):
            _cls, _reg, _centerness = FCOSLoss.create_target_for_level(
                bbox, 
                cls, 
                self.num_classes, 
                stride, 
                size_target, 
                irange, 
                self.radius
            )
            class_targets.append(_cls)
            reg_targets.append(_reg)
            centerness_targets.append(_centerness)

        return tuple([
            (cls, reg, centerness) 
            for cls, reg, centerness in zip(class_targets, reg_targets, centerness_targets)
        ])

    @staticmethod
    def create_target_for_level(
        bbox: Tensor, 
        cls: Tensor, 
        num_classes: int,
        stride: int,
        size_target: Tuple[int, int],
        interest_range: Tuple[int, int],
        center_radius: Optional[int] = None
    ) -> Tensor:
        # get bbox locations within feature map after stride is applied
        bbox_stride = bbox.floor_divide(stride)

        # build regression target
        reg_target = FCOSLoss.create_regression_target(bbox_stride, stride, size_target)

        # use the regression targets to determine boxes of interest for this level
        # is of interest if lower_bound <= max(l, r, t, b) <= upper_bound
        max_size = reg_target.view(reg_target.shape[0], -1).max(dim=-1).values
        lower_bound, upper_bound = interest_range
        is_box_of_interest = (max_size >= lower_bound).logical_and_(max_size <= upper_bound)

        # get mask of valid locations within each box and apply boxes_of_interest filter
        mask = FCOSLoss.bbox_to_mask(bbox, stride, size_target, center_radius)
        mask[~is_box_of_interest] = False

        # build classification target
        cls_target = FCOSLoss.create_classification_target(bbox, cls, mask, num_classes, size_target)

        # apply mask to regression target and take per pixel maximum for all boxes
        reg_target[~mask[..., None, :, :].expand_as(reg_target)] = IGNORE
        reg_target = reg_target.max(dim=0).values

        centerness_target = FCOSLoss.compute_centerness_targets(reg_target)
        centerness_target[~mask.any(dim=-3, keepdim=True)] = IGNORE

        return cls_target, reg_target, centerness_target


    @staticmethod
    def bbox_to_mask(
        bbox: Tensor, 
        stride: int,
        size_target: Tuple[int, int],
        center_radius: Optional[float] = None
    ) -> Tensor:
        check_is_tensor(bbox, "bbox")
        check_dimension(bbox, -1, 4, "bbox")

        # create empty masks
        num_boxes = bbox.shape[-2]
        h = torch.arange(size_target[0], dtype=bbox.dtype, device=bbox.device)
        w = torch.arange(size_target[1], dtype=bbox.dtype, device=bbox.device)
        mask = torch.stack(torch.meshgrid(h, w), 0).unsqueeze_(0).expand(num_boxes, -1, -1, -1)

        # get edge coordinates of each box based on whole box or center sampled
        lower_bound = bbox[..., :2]
        upper_bound = bbox[..., 2:]
        if center_radius is not None:
            # update bounds according to radius from center
            center = (bbox[..., :2] + bbox[..., 2:]).floor_divide_(2)
            offset = torch.tensor([stride, stride], device=bbox.device, dtype=center.dtype).mul_(center_radius)
            lower_bound = (center - offset[None])
            upper_bound = (center + offset[None])

        # x1y1 to h1w1, add h/w dimensions, convert to strided coords
        lower_bound = lower_bound[..., (1, 0), None, None].floor_divide_(stride)
        upper_bound = upper_bound[..., (1, 0), None, None].floor_divide_(stride)

        # use edge coordinates to create a binary mask
        mask = (mask > lower_bound).logical_and_(mask < upper_bound).all(dim=-3)
        return mask

    @staticmethod
    def create_regression_target(
        bbox: Tensor, 
        stride: int,
        size_target: Tuple[int, int],
    ) -> Tensor:
        check_is_tensor(bbox, "bbox")
        check_dimension(bbox, -1, 4, "bbox")

        # create starting grid
        num_boxes = bbox.shape[-2]
        h = torch.arange(size_target[0], dtype=bbox.dtype, device=bbox.device)
        w = torch.arange(size_target[1], dtype=bbox.dtype, device=bbox.device)
        grid = torch.meshgrid(h, w)
        grid = torch.stack([grid[1], grid[0]], dim=0).unsqueeze_(0).repeat(num_boxes, 2, 1, 1)
        grid.mul_(stride)

        # compute distance to box edges relative to each grid location
        grid[..., :2, :, :].sub_(bbox[..., :2, None, None])
        grid[..., 2:, :, :].neg_().add_(bbox[..., 2:, None, None])
        return grid

    @staticmethod
    def create_classification_target(
        bbox: Tensor, 
        cls: Tensor,
        mask: Tensor,
        num_classes: int,
        size_target: Tuple[int, int],
    ) -> Tensor:
        check_is_tensor(bbox, "bbox")
        check_is_tensor(cls, "cls")
        check_is_tensor(mask, "mask")
        check_dimension_match(bbox, cls, -2, "bbox", "cls")
        check_dimension_match(bbox, mask, 0, "bbox", "mask")
        check_dimension(bbox, -1, 4, "bbox")
        check_dimension(cls, -1, 1, "cls")

        target = torch.zeros(num_classes, *mask.shape[-2:], device=mask.device, dtype=torch.float)

        box_id, h, w = mask.nonzero(as_tuple=True)
        class_id = cls[box_id, 0]
        target[class_id, h, w] = 1.0
        return target

    @staticmethod
    def compute_centerness_targets(reg_targets: Tensor) -> Tensor:
        r"""Computes centerness targets given a 2D map of regression targets. 

        Under FCOS, a target regression map is created for each FPN level. Any map location
        that lies within a ground truth bounding box is assigned a regression target based on
        the left, right, top, and bottom distance from that location to the edges of the ground
        truth box.

        .. image:: ./fcos_target.png
            :width: 200px
            :align: center
            :height: 600px
            :alt: FCOS Centerness Target

        For each of these locations with regression targets :math:`l^*, r^*, t^*, b^*`, 
        a "centerness" target is created as follows:

        .. math::
            centerness = \sqrt{\frac{\min(l^*, r*^}{\max(l^*, r*^} \times \frac{\min(t^*, b*^}{\max(t^*, b*^}}

        Args: 
            reg_targets (:class:`torch.Tensor`):
                Ground truth regression featuremap in form :math:`x_1, y_1, x_2, y_2`.

        Shapes:
            * ``reg_targets`` - :math:`(..., 4, H, W)`
            * Output - :math:`(..., 1, H, W)`
        """
        check_is_tensor(reg_targets, "reg_targets")
        check_dimension(reg_targets, -3, 4, "reg_targets")

        left_right = reg_targets[(0, 2), ...].float()
        top_bottom = reg_targets[(1, 3), ...].float()

        lr_min = left_right.min(dim=-3).values.clamp_min_(0)
        lr_max = left_right.max(dim=-3).values.clamp_min_(1) 
        tb_min = top_bottom.min(dim=-3).values.clamp_min_(0)
        tb_max = top_bottom.max(dim=-3).values.clamp_min_(1)

        centerness_lr = lr_min.true_divide_(lr_max)
        centerness_tb = tb_min.true_divide_(tb_max)
        centerness = centerness_lr.mul_(centerness_tb).sqrt_().unsqueeze_(-3)

        assert centerness.shape[-2:] == reg_targets.shape[-2:]
        assert centerness.shape[-3] == 1
        assert centerness.ndim == reg_targets.ndim
        return centerness

from typing import Iterable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import ByteTensor, Tensor

from combustion.util import alpha_blend, apply_colormap, check_is_tensor, check_ndim_match

from .convert import to_8bit
from .iou_assign import CategoricalLabelIoU



class FCOSMixin(CenterNetMixin):
    PAD_VALUE: float = -1

    @staticmethod
    def split_box_target(target: Tensor, split_label: Union[bool, Iterable[int]] = False) -> Tuple[Tensor, ...]:
        r"""Split a bounding box label set into box coordinates and label tensors.

        .. note::
            This operation returns views of the original tensor.

        Args:
            target (:class:`torch.Tensor`):
                The target to split.

            split_label (bool or iterable of ints):
                Whether to further decompose the label tensor. If ``split_label`` is ``True``, split
                the label tensor along the last dimension. If an interable of ints is given, treat each
                int as a split size arugment to :func:`torch.split` along the last dimension.

        Shape:
            * ``target`` - :math:`(*, N, 4 + C)` where :math:`N` is the number of boxes and :math:`C` is the
              number of labels associated with each box.

            * Output - :math:`(*, N, 4)` and :math:`(*, N, C)`
        """
        check_is_tensor(target, "target")
        bbox = target[..., :4]
        label = target[..., 4:]

        if isinstance(split_label, bool) and not split_label:
            return bbox, label

        num_labels = label.shape[-1]

        # setup split size of 1 if bool given
        if isinstance(split_label, bool):
            split_label = [
                1,
            ] * num_labels

        lower_bound = 0
        upper_bound = 0
        final_label = []
        for delta in split_label:
            upper_bound = lower_bound + delta
            final_label.append(label[..., lower_bound:upper_bound])
            lower_bound = upper_bound
        assert len(final_label) == len(split_label)
        return tuple([bbox] + final_label)

    @staticmethod
    def split_bbox_scores_class(target: Tensor, split_scores: Union[bool, Iterable[int]] = False) -> Tuple[Tensor, ...]:
        r"""Split a predicted bounding box into box coordinates, probability score, and predicted class.
        This implementation supports multiple score assignments for each box. It is expected that ``target``
        be ordered along the last dimension as ``bbox``, ``scores``, ``class``.

        .. note::
            This operation returns views of the original tensor.

        Args:
            target (:class:`torch.Tensor`):
                The target to split.

            split_scores (bool or iterable of ints):
                Whether to further decompose the scores tensor. If ``split_scores`` is ``True``, split
                the scores tensor along the last dimension. If an interable of ints is given, treat each
                int as a split size arugment to :func:`torch.split` along the last dimension.

        Shape:
            * ``target`` - :math:`(*, N, 4 + S + 1)` where :math:`N` is the number of boxes and :math:`S` is the
              number of scores associated with each box.
            * Output - :math:`(*, N, 4)`, :math:`(*, N, S)`, and :math:`(*, N, 1)`
        """
        check_is_tensor(target, "target")
        bbox = target[..., :4]
        scores = target[..., 4:-1]
        cls = target[..., -1:]

        if isinstance(split_scores, bool) and not split_scores:
            return bbox, scores, cls

        num_scores = scores.shape[-1]

        # setup split size of 1 if bool given
        if isinstance(split_scores, bool):
            split_scores = [
                1,
            ] * num_scores

        lower_bound = 0
        upper_bound = 0
        final_scores = []
        for delta in split_scores:
            upper_bound = lower_bound + delta
            final_scores.append(scores[..., lower_bound:upper_bound])
            lower_bound = upper_bound
        assert len(final_scores) == len(split_scores)
        return tuple([bbox] + final_scores + [cls])

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
    def combine_box_target(bbox: Tensor, label: Tensor, *extra_labels) -> Tensor:
        r"""Combine a bounding box coordinates and labels into a single label.

        Args:
            bbox (:class:`torch.Tensor`):
                Coordinates of the bounding box.

            label (:class:`torch.Tensor`):
                Label associated with each bounding box.

        Shape:
            * ``bbox`` - :math:`(*, N, 4)`
            * ``label`` - :math:`(*, N, 1)`
            * Output - :math:`(*, N, 4 + 1)`
        """
        # input validation
        check_is_tensor(bbox, "bbox")
        check_is_tensor(label, "label")
        if bbox.shape[-1] != 4:
            raise ValueError(f"Expected bbox.shape[-1] == 4, found shape {bbox.shape}")
        if bbox.shape[:-1] != label.shape[:-1]:
            raise ValueError(f"Expected bbox.shape[:-1] == label.shape[:-1], found shapes {bbox.shape}, {label.shape}")

        return torch.cat([bbox, label, *extra_labels], dim=-1)

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
    def batch_box_target(target: List[Tensor], pad_value: float = -1) -> Tensor:
        r"""Combine multiple distinct bounding box targets into a single batched target.

        Args:
            target (list of :class:`torch.Tensor`):
                List of bounding box targets to combine

            pad_value (float):
                Padding value to use when creating the batch

        Shape:
            * ``target`` - :math:`(*, N_i, 4 + C)` where :math:`N_i` is the number of boxes and :math:`C` is the
              number of labels associated with each box.

            * Output - :math:`(B, N, 4 + c)`
        """
        max_boxes = 0
        for elem in target:
            check_is_tensor(elem, "target_elem")
            max_boxes = max(max_boxes, elem.shape[-2])

        # add a batch dim if not present
        target = [x.view(1, *x.shape) if x.ndim < 3 else x for x in target]

        # compute output batch size
        batch_size = sum([x.shape[0] for x in target])

        # create empty output tensor of correct shape
        output_shape = (batch_size, max_boxes, target[0].shape[-1])
        batch = torch.empty(*output_shape, device=target[0].device, dtype=target[0].dtype).fill_(pad_value)

        # fill output tensor
        start = 0
        for elem in target:
            end = start + elem.shape[0]
            batch[start:end, : elem.shape[-2], :] = elem
            start += elem.shape[0]

        return batch

    @staticmethod
    def unbatch_box_target(target: Tensor, pad_value: float = -1) -> List[Tensor]:
        r"""Splits a padded batch of bounding boxtarget tensors into a list of unpadded target tensors

        Args:
            target (:class:`torch.Tensor`):
                Batch of bounding box targets to split

            pad_value (float):
                Value used for padding when creating the batch

        Shape:
            * ``target`` - :math:`(B, N, 4 + C)` where :math:`N` is the number of boxes and :math:`C` is the
              number of labels associated with each box.

            * Output - :math:`(N, 4 + C)`
        """
        check_is_tensor(target, "target")

        padding_indices = (target == pad_value).all(dim=-1)
        non_padded_coords = (~padding_indices).nonzero(as_tuple=True)

        flat_result = target[non_padded_coords]
        split_size = non_padded_coords[0].unique(return_counts=True)[1]
        return torch.split(flat_result, split_size.tolist(), dim=0)

    @staticmethod
    def get_pred_target_pairs(
        pred: Tensor,
        target: Tensor,
        upsample: int,
        iou_threshold: float = 0.5,
        true_positive_limit: bool = True,
        pad_value: float = -1,
    ) -> Tensor:
        r"""Given a predicted CenterNet heatmap and target bounding box label, use box IoU to
        create a paring of predicted and target boxes such that each predicted box has
        an associated gold standard label.

        .. warning::
            This method should work with batched input, but such inputs are not thoroughly tested

        Args:
            pred (:class:`torch.Tensor`):
                Predicted heatmap.

            target (:class:`torch.Tensor`):
                Target bounding boxes in format ``x1, y1, x2, y2, class``.

            iou_threshold (float):
                Intersection over union threshold for which a prediction can be considered a
                true positive.

            true_positive_limit (bool):
                By default, only one predicted box overlapping a target box will be counted
                as a true positive. If ``False``, allow multiple true positive boxes per
                target box.

            pad_value (float):
                Value used for padding a batched input, and the value to use when padding
                a batched output.

        Returns:
            Tensor paring a predicted probability with an integer class label. If input is a batch,
            return a list with result tensors for each batch element.

        Shape:
            * ``pred`` - :math:`(*, C+4, H, W)`
            * ``target`` - :math:`(*, N_i, 5)`
            * Output - :math:`(*, N_o, 2)`
        """
        check_is_tensor(pred, "pred")
        check_is_tensor(target, "target")
        is_batch = pred.ndim > 3
        assert pred.shape[-3] > 4

        if is_batch:
            assert target.ndim > 2, "pred batched but target not batched"
            batch_result = []
            for pred_i, target_i in zip(pred, target):
                result = CenterNetMixin.get_pred_target_pairs(
                    pred_i, target_i, upsample, iou_threshold, true_positive_limit, pad_value
                )
                batch_result.append(result)
            return batch_result

        # we might be operating on a batched example that was padded, so remove these padded locations
        pad_locations = (target == -1).all(dim=-1)
        target = target[~pad_locations, :]

        # turn heatmap into anchor boxes with no threshold / max roi
        # this generates all boxes that satisfy the > 8 nearest neighbors condition
        xform = PointsToAnchors(upsample, max_roi=None, threshold=0.0)
        pred = xform(pred)

        # get a paring of predicted probability to target labels
        # if we didn't detect a target box at any threshold, assume P_pred = 0.0
        xform = CategoricalLabelIoU(iou_threshold, true_positive_limit)
        pred_boxes, pred_scores, pred_cls = CenterNetMixin.split_bbox_scores_class(pred)
        target_bbox, target_cls = CenterNetMixin.split_box_target(target)
        pred_out, is_correct, target_out = xform(pred_boxes, pred_scores, pred_cls, target_bbox, target_cls)

        assert pred_out.ndim == 1
        assert target_out.ndim == 1
        assert pred_out.shape == target_out.shape
        return pred_out, target_out.long(), is_correct

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

