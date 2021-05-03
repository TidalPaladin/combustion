#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor

from combustion.util import check_dimension, check_dimension_match, check_is_tensor

from .ciou import CompleteIoULoss
from .focal import FocalLossWithLogits


IGNORE = -1
INF = 10000000

# FCOS uses each FPN level to predict targets of different sizes
# This is the size range used in the paper
DEFAULT_INTEREST_RANGE: Tuple[Tuple[int, int], ...] = (
    (-1, 64),  # stride=8
    (64, 128),  # stirde=16
    (128, 256),  # stride=32
    (256, 512),  # stride=64
    (512, 10000000),  # stride=128
)


class FCOSLoss:
    r"""Implements the loss function and target creation as described in PLACEHOLDER.

    Args:
        strides (tuple of ints):
            Stride at each FCOS FPN level.

        num_classes (int):
            Number of classes being detected

        interest_range (tuple of (int, int)):
            Lower and upper bound on target object sizes for each FPN level

        gamma (float):
            Gamma term for focal loss. See :class:`combustion.nn.FocalLossWithLogits`

        alpha (float):
            Alpha (positive example weight) term for focal loss.
            See :class:`combustion.nn.FocalLossWithLogits`

        radius (int):
            Radius (in stride units) about box centers for which heatmap locations
            should be considered positive examples.

        pad_value (float):
            Padding value when batching / unbatching anchor boxes

    Returns:
        Tuple of (``cls_loss``, ``reg_loss``, ``centerness_loss``)

    Shapes:
        * ``cls_pred`` - :math:`(N, C, H_i, W_i)` where :math:`i` is the :math:`i`'th FPN level
        * ``reg_pred`` - :math:`(N, 4, H_i, W_i)`
        * ``centerness_pred`` - :math:`(N, 1, H_i, W_i)`
        * ``target_bbox`` - :math:`(N, B, 4)`
        * ``target_cls`` - :math:`(N, B, 1)`
    """

    def __init__(
        self,
        strides: Tuple[int, ...],
        num_classes: int,
        interest_range: Tuple[Tuple[int, int], ...] = DEFAULT_INTEREST_RANGE,
        gamma: float = 2.0,
        alpha: float = 0.5,
        radius: Optional[int] = 1,
        pad_value: float = -1,
    ):
        self.strides = tuple([int(x) for x in strides])
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
        target_cls: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        batch_size = target_bbox.shape[0]
        cls_loss, reg_loss, centerness_loss = [], [], []

        for i in range(batch_size):
            cls = [t[i] for t in cls_pred]
            reg = [t[i] for t in reg_pred]
            centerness = [t[i] for t in centerness_pred]
            padding = (target_bbox[i] == self.pad_value).all(dim=-1)
            _target_bbox = target_bbox[i][~padding]
            _target_cls = target_cls[i][~padding]
            assert _target_bbox.shape[:-1] == _target_cls.shape[:-1]
            _cls_loss, _reg_loss, _centerness_loss = self.compute_from_box_target(
                cls, reg, centerness, _target_bbox, _target_cls
            )
            cls_loss.append(_cls_loss)
            reg_loss.append(_reg_loss)
            centerness_loss.append(_centerness_loss)

        cls_loss = sum(cls_loss) / batch_size
        reg_loss = sum(reg_loss) / batch_size
        centerness_loss = sum(centerness_loss) / batch_size
        return cls_loss, reg_loss, centerness_loss

    def _reduce(self, cls_loss, reg_loss, centerness_loss, fcos_target):
        num_gpus = self.get_num_gpus()
        cls_target, reg_target, centerness_target = list(zip(*fcos_target))
        pos_inds = sum([x.sum() for x in cls_target])
        centerness_inds = sum([(x != IGNORE).sum() for x in centerness_target])

        total_num_pos = self.reduce_sum(pos_inds).item()
        num_pos_avg_per_gpu = max(total_num_pos / float(num_gpus), 1.0)
        sum_centerness_targets_avg_per_gpu = self.reduce_sum(centerness_inds).item() / float(num_gpus)

        cls_loss = sum([x.sum() for x in cls_loss])
        reg_loss = sum([x.sum() for x in reg_loss])
        centerness_loss = sum([x.sum() for x in centerness_loss])

        cls_loss = cls_loss / max(num_pos_avg_per_gpu, 1.0)
        reg_loss = reg_loss / max(sum_centerness_targets_avg_per_gpu, 1.0)
        centerness_loss = centerness_loss / max(num_pos_avg_per_gpu, 1.0)
        return cls_loss, reg_loss, centerness_loss

    def get_num_gpus(self) -> int:
        return int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    def reduce_sum(self, tensor: Tensor) -> Tensor:
        if self.get_num_gpus() <= 1:
            return tensor
        tensor = tensor.clone()
        dist.all_reduce(tensor, op=dist.reduce_op.SUM)
        return tensor

    def _drop_padding(self, x: Tensor) -> Tensor:
        padding = (x == self.pad_value).all(dim=-1)
        return x[~padding]

    def compute_from_box_target(
        self,
        cls_pred: Tuple[Tensor, ...],
        reg_pred: Tuple[Tensor, ...],
        centerness_pred: Tuple[Tensor, ...],
        target_bbox: Tensor,
        target_cls: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        size_targets = tuple([x.shape[-2:] for x in cls_pred])
        with torch.no_grad():
            fcos_targets = self.create_targets(target_bbox, target_cls, size_targets)
        return self.compute_from_fcos_target(cls_pred, reg_pred, centerness_pred, fcos_targets)

    def compute_from_fcos_target(
        self,
        cls_pred: Tuple[Tensor, ...],
        reg_pred: Tuple[Tensor, ...],
        centerness_pred: Tuple[Tensor, ...],
        fcos_target: Tuple[Tuple[Tensor, Tensor, Tensor], ...],
    ) -> Tuple[Tensor, Tensor, Tensor]:

        cls_loss, reg_loss, centerness_loss = [], [], []
        z = zip(cls_pred, reg_pred, centerness_pred, fcos_target)
        for cls_pred_i, reg_pred_i, centerness_pred_i, (cls_true, reg_true, centerness_true) in z:
            _cls_loss = self.cls_criterion(cls_pred_i, cls_true)

            reg_ignore = (reg_true == IGNORE).all(dim=-3)
            centerness_ignore = centerness_true == IGNORE

            centerness_pred_i = centerness_pred_i[~centerness_ignore]
            reg_pred_i = reg_pred_i[:, ~reg_ignore].view(4, -1)
            centerness_true = centerness_true[~centerness_ignore]
            reg_true = reg_true[:, ~reg_ignore].view(4, -1)

            _reg_loss = self.reg_criterion(reg_pred_i.permute(1, 0), reg_true.permute(1, 0))
            _centerness_loss = self.centerness_criterion(centerness_pred_i, centerness_true)

            cls_loss.append(_cls_loss)
            reg_loss.append(_reg_loss)
            centerness_loss.append(_centerness_loss)

        cls_loss, reg_loss, centerness_loss = self._reduce(cls_loss, reg_loss, centerness_loss, fcos_target)
        return cls_loss, reg_loss, centerness_loss

    def create_targets(
        self, bbox: Tensor, cls: Tensor, size_targets: Tuple[Tuple[int, int], ...]
    ) -> Tuple[Tuple[Tensor, Tensor, Tensor], ...]:
        if bbox.ndim >= 3:
            batch_size = bbox.shape[0]
            targets = []
            for i in range(batch_size):
                bbox_i = bbox[i]
                cls_i = cls[i]
                t = self.create_targets(bbox_i, cls_i, size_targets)
                targets.append(t)

            cls_targets, reg_targets, centerness_targets = [], [], []
            for level in range(len(size_targets)):
                targets_for_level = [item[level] for item in targets]
                cls_targets_i = torch.stack([tar[0] for tar in targets_for_level], dim=0)
                reg_targets_i = torch.stack([tar[1] for tar in targets_for_level], dim=0)
                centerness_targets_i = torch.stack([tar[2] for tar in targets_for_level], dim=0)
                cls_targets.append(cls_targets_i)
                reg_targets.append(reg_targets_i)
                centerness_targets.append(centerness_targets_i)

            return cls_targets, reg_targets, centerness_targets

        class_targets, reg_targets, centerness_targets = [], [], []
        for irange, stride, size_target in zip(self.interest_range, self.strides, size_targets):
            _cls, _reg, _centerness = FCOSLoss.create_target_for_level(
                bbox, cls, self.num_classes, stride, size_target, irange, self.radius
            )
            class_targets.append(_cls)
            reg_targets.append(_reg)
            centerness_targets.append(_centerness)

        return tuple(
            [(cls, reg, centerness) for cls, reg, centerness in zip(class_targets, reg_targets, centerness_targets)]
        )

    @staticmethod
    def create_target_for_level(
        bbox: Tensor,
        cls: Tensor,
        num_classes: int,
        stride: int,
        size_target: Tuple[int, int],
        interest_range: Tuple[int, int],
        center_radius: Optional[int] = None,
    ) -> Tensor:
        # handle case of no boxes
        if not bbox.numel():
            cls_target = torch.zeros(num_classes, *size_target, device=cls.device, dtype=torch.float)
            reg_target = bbox.new_empty(4, *size_target).fill_(-1)
            centerness_target = cls_target.new_empty(1, *size_target).fill_(-1)
            return cls_target, reg_target, centerness_target

        # get mask of valid locations within each box and apply boxes_of_interest filter
        inside_box_mask = FCOSLoss.bbox_to_mask(bbox, stride, size_target)
        mask = FCOSLoss.bbox_to_mask(bbox, stride, size_target, center_radius)

        # build regression target
        reg_target = FCOSLoss.create_regression_target(bbox, stride, size_target)
        reg_target[~inside_box_mask[..., None, :, :].expand_as(reg_target)] = IGNORE

        # use the regression targets to determine boxes of interest for this level
        # is of interest if lower_bound <= max(l, r, t, b) <= upper_bound
        bounds = bbox.new_tensor(interest_range).unsqueeze_(0)
        is_box_of_interest = FCOSLoss.assign_boxes_to_levels(bbox, bounds, inclusive="lower").squeeze_(-1)

        if not is_box_of_interest.any():
            reg_target.fill_(IGNORE)
            centerness = torch.empty_like(reg_target[..., 0:1, :, :]).fill_(IGNORE)
            cls_target = centerness.new_zeros(num_classes, *centerness.shape[-2:])

        mask[~is_box_of_interest] = False
        inside_box_mask[~is_box_of_interest] = False
        reg_target[~is_box_of_interest] = IGNORE

        # build classification target
        cls_target = FCOSLoss.create_classification_target(bbox, cls, mask, num_classes, size_target)

        # apply mask to regression target and take per pixel minimum for all boxes
        reg_target[reg_target == IGNORE] = INF
        reg_target = reg_target.amin(dim=-4)
        reg_target[reg_target == INF] = IGNORE

        # build centerness target
        centerness_target = torch.empty_like(reg_target[..., 0:1, :, :]).fill_(IGNORE)
        ind = (reg_target == IGNORE).all(dim=-3).logical_not_()
        if ind.any():
            pos_reg_targets = reg_target[..., ind].permute(1, 0)
            centerness_target[..., ind] = FCOSLoss.compute_centerness_targets(pos_reg_targets).view(-1)

        reg_target[:, ~mask.any(dim=-3)] = IGNORE

        return cls_target, reg_target, centerness_target

    @staticmethod
    def bbox_to_mask(
        bbox: Tensor, stride: int, size_target: Tuple[int, int], center_radius: Optional[float] = None
    ) -> Tensor:
        r"""Creates a mask for each input anchor box indicating which heatmap locations for that
        box should be positive examples. Under FCOS, a target maps are created for each FPN level.
        Any map location that lies within ``center_radius * stride`` units from the center of the
        ground truth bounding box is considered a positive example for regression and classification.

        This method creates a mask for FPN level with stride ``stride``. The mask will have shape
        :math:`(N, H, W)` where :math:`(H, W)` are given in ``size_target``. Mask locations that
        lie within ``center_radius * stride`` units of the box center will be ``True``. If
        ``center_radius=None``, all locations within a box will be considered positive.

        Args:
            bbox (:class:`torch.Tensor`):
                Ground truth anchor boxes in form :math:`x_1, y_1, x_2, y_2`.

            stride (int):
                Stride at the FPN level for which the target is being created

            size_target (tuple of int, int):
                Height and width of the mask. Should match the height and width of the FPN
                level for which a target is being created.

            center_radius (float, optional):
                Radius (in units of ``stride``) about the center of each box for which examples
                should be considered positive. If ``center_radius=None``, all locations within
                a box will be considered positive.

        Shapes:
            * ``reg_targets`` - :math:`(..., 4, H, W)`
            * Output - :math:`(..., 1, H, W)`
        """
        check_is_tensor(bbox, "bbox")
        check_dimension(bbox, -1, 4, "bbox")

        # create mesh grid of size `size_target`
        # locations in grid give h/w at center of that location
        #
        # we will compare bbox coords against this grid to find locations that lie within
        # the center_radius of bbox
        num_boxes = bbox.shape[-2]
        h = torch.arange(size_target[0], dtype=torch.float, device=bbox.device)
        w = torch.arange(size_target[1], dtype=torch.float, device=bbox.device)
        mask = (
            torch.stack(torch.meshgrid(h, w), 0)
            .mul_(stride)
            .add_(stride / 2)
            .unsqueeze_(0)
            .expand(num_boxes, -1, -1, -1)
        )

        # get edge coordinates of each box based on whole box or center sampled
        lower_bound = bbox[..., :2]
        upper_bound = bbox[..., 2:]
        if center_radius is not None:
            assert center_radius >= 1
            # update bounds according to radius from center
            center = (bbox[..., :2] + bbox[..., 2:]).true_divide(2)
            offset = center.new_tensor([stride, stride]).mul_(center_radius)
            lower_bound = torch.max(lower_bound, center - offset[None])
            upper_bound = torch.min(upper_bound, center + offset[None])

        # x1y1 to h1w1, add h/w dimensions, convert to strided coords
        lower_bound = lower_bound[..., (1, 0), None, None]
        upper_bound = upper_bound[..., (1, 0), None, None]

        # use edge coordinates to create a binary mask
        mask = (mask >= lower_bound).logical_and_(mask <= upper_bound).all(dim=-3)
        return mask

    @staticmethod
    def create_regression_target(bbox: Tensor, stride: int, size_target: Tuple[int, int]) -> Tensor:
        r"""Given a set of anchor boxes, creates regression targets each anchor box.
        Each location in the resultant target gives the distance from that location to the
        left, top, right, and bottom of the ground truth anchor box (in that order).

        Args:
            bbox (:class:`torch.Tensor`):
                Ground truth anchor boxes in form :math:`x_1, y_1, x_2, y_2`.

            stride (int):
                Stride at the FPN level for which the target is being created

        Shapes:
            * ``bbox`` - :math:`(*, N, 4)`
            * Output - :math:`(*, N, 2)`, :math:`(*, N, 4)`
        """
        check_is_tensor(bbox, "bbox")
        check_dimension(bbox, -1, 4, "bbox")

        # create starting grid

        num_boxes = bbox.shape[-2]
        height, width = size_target[0], size_target[1]
        grid = FCOSLoss.coordinate_grid(height, width, stride, indexing="xy", device=bbox.device)
        grid = grid.unsqueeze_(0).repeat(num_boxes, 2, 1, 1)

        # compute distance to box edges relative to each grid location
        grid.sub_(bbox[..., None, None]).abs_()
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
        r"""Computes centerness targets given regression targets.

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
            * ``reg_targets`` - :math:`(..., 4)`
            * Output - :math:`(..., 1)`
        """
        check_is_tensor(reg_targets, "reg_targets")
        check_dimension(reg_targets, -1, 4, "reg_targets")

        left_right = reg_targets[..., (0, 2)].float()
        top_bottom = reg_targets[..., (1, 3)].float()

        lr_min = left_right.amin(dim=-1).clamp_min_(0)
        lr_max = left_right.amax(dim=-1).clamp_min_(1)
        tb_min = top_bottom.amin(dim=-1).clamp_min_(0)
        tb_max = top_bottom.amax(dim=-1).clamp_min_(1)

        centerness_lr = lr_min.true_divide_(lr_max)
        centerness_tb = tb_min.true_divide_(tb_max)
        centerness = centerness_lr.mul_(centerness_tb).sqrt_().unsqueeze_(-1)

        assert centerness.shape[:-1] == reg_targets.shape[:-1]
        assert centerness.shape[-1] == 1
        assert centerness.ndim == reg_targets.ndim
        return centerness

    @staticmethod
    def get_box_center(bbox: Tensor) -> Tensor:
        return bbox.view(-1, 2, 2).sum(dim=-2).float().div_(2)

    @staticmethod
    @torch.jit.script
    def coordinate_grid(
        height: int, width: int, stride: int = 1, indexing: str = "hw", device: Optional[torch.device] = None
    ) -> Tensor:
        r"""Creates a coordinate grid of a given height and width where each location
        This is used to map locations in a FCOS predication at a given stride.

        Args:
            height (int):
                Height of the resultant grid

            width (int):
                Width of the resultant grid

            stride (int):
                Step size between adjacent locations in the grid

            indexing (str):
                One of ``"hw"`` or ``"xy"``. If ``xy``, each coordinate pair in the grid
                describes a coordinate by x and y. Otherwise, each coordinate
                pair describes a coordinate by height and width.

            device (:class:`torch.device`):
                Device for the resultant grid to be placed on

        Shape:
            * Output - :math:`(2, H, W)`

        Example:
            >>> # Assume we have a FCOS prediction at stride 16.
            >>> # We can map a positive location to a position in the
            >>> # original image as follows...
            >>>
            >>> # positive prediction at location 0, 0
            >>> cls_pred = torch.zeros(1, 1, 32, 32)
            >>> cls_pred[..., 4, 4] = 1.0
            >>>
            >>> # positive prediction at location 0, 0
            >>> grid = FCOSLoss.coordinate_grid(32, 32, stride=16)
            >>> box_center_hw = grid[..., 4, 4] # center at 4 * 16 + 16 / 2
        """
        h = torch.arange(height, dtype=torch.float, device=device)
        w = torch.arange(width, dtype=torch.float, device=device)
        grid_h, grid_w = torch.meshgrid(h, w)

        if indexing == "hw":
            grid = torch.stack((grid_h, grid_w), 0)
        elif indexing == "xy":
            grid = torch.stack((grid_w, grid_h), 0)
        else:
            raise ValueError(f"Invalid indexing: {indexing}")

        return grid.mul_(stride).add_(stride / 2)

    @staticmethod
    @torch.jit.script
    def assign_boxes_to_levels(bbox: Tensor, bounds: Tensor, inclusive: str = "lower") -> Tensor:
        r"""Given bounding boxes and upper/lower size threshold for each FPN level,
        determine assignments of boxes to FPN levels. Boxes are assigned to a FPN level
        if the longest box edge falls between the upper and lower bound for that level.

        Args:
            bbox (:class:`torch.Tensor`):
                Bounding boxes to be assigned to levels in order :math:`x_1, y_1, x_2, y_2`

            bounds (:class:`torch.Tensor`):
                Lower and upper size interest ranges for each FPN level

            inclusive (str):
                Determines if comparisons at the lower and upper bounds are inclusive or
                exclusive. Should be one of ``"lower"``, ``"upper"``, ``"both"``.

        Returns:
            Bool tensor indicating which level(s) a box is assigned to

        Shape:
            * ``bbox`` - :math:`(*, N, 4)` in :math:`x_1, y_1, x_2, y_2` order
            * ``bounds`` - :math:`(B, 2)` in :math:`lower, upper` order
            * Output - :math:`(*, N, B)`
        """
        # check_is_tensor(bbox, "bbox")
        # check_is_tensor(bounds, "bounds")
        # check_dimension(bounds, -1, 2, "bounds")
        # check_dimension(bbox, -1, 4, "bbox")
        inclusive = str(inclusive).lower()
        if inclusive not in ("lower", "upper", "both"):
            raise ValueError(f"Invalid value for `inclusive`: {inclusive}")

        box_low, box_high = torch.split(bbox, 2, dim=-1)
        bound_low, bound_high = bounds[..., 0], bounds[..., 1]

        longest_edge = (box_high - box_low).amax(dim=-1, keepdim=True)

        if inclusive in ("lower", "both"):
            mask = longest_edge >= bound_low
        else:
            mask = longest_edge > bound_low

        if inclusive in ("upper", "both"):
            mask.logical_and_(longest_edge <= bound_high)
        else:
            mask.logical_and_(longest_edge < bound_high)

        assert mask.shape[:-1] == bbox.shape[:-1]
        assert mask.shape[-1] == bounds.shape[-2]
        return mask
