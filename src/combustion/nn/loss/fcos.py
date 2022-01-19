#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from typing import Iterable, Optional, Tuple, Union, List, Iterator

import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor

from combustion.util import check_dimension, check_dimension_match, check_is_tensor
from combustion.util.dataclasses import TensorDataclass
from dataclasses import dataclass

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


def create_regression_target(bbox: Tensor, stride: int, size_target: Tuple[int, int]) -> Tensor:
    r"""Given a set of anchor boxes, creates regression targets for each anchor box.
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
    B, N, C = bbox.shape
    H, W = size_target

    # create starting grid

    num_boxes = bbox.shape[-2]
    height, width = size_target[0], size_target[1]
    grid = coordinate_grid(height, width, stride, indexing="xy", device=bbox.device)
    grid = grid.view(1, 1, 2, H, W).repeat(B, num_boxes, 2, 1, 1)
    assert grid.shape == (B, N, 4, H, W)

    # compute distance to box edges relative to each grid location
    bbox = bbox.view(B, N, 4, 1, 1)
    grid.sub_(bbox).abs_()
    return grid


def create_classification_target(
    bbox: Tensor,
    cls: Tensor,
    mask: Tensor,
    num_classes: int,
) -> Tensor:
    B, N, _ = bbox.shape
    _, _, H, W = mask.shape
    target = torch.zeros(B, num_classes, H, W, device=mask.device, dtype=torch.float)
    batch_idx, box_id, h, w = mask.nonzero(as_tuple=True)
    class_id = cls[batch_idx, box_id, 0]
    target[batch_idx, class_id, h, w] = 1.0
    return target


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



def get_num_gpus() -> int:
    return int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

def reduce_sum(tensor: Tensor) -> Tensor:
    if get_num_gpus() <= 1:
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.reduce_op.SUM)
    return tensor


@dataclass(repr=False)
class FCOSLevelPrediction(TensorDataclass):
    cls: Tensor
    reg: Tensor
    centerness: Tensor


@dataclass(repr=False)
class FCOSPrediction(TensorDataclass):
    preds: Tuple[FCOSLevelPrediction, ...]

    def __iter__(self) -> Iterator[FCOSLevelPrediction]:
        for t in self.preds:
            yield t

    def __getitem__(self, level: int) -> FCOSLevelPrediction:
        return self.preds[level]

    def __len__(self) -> int:
        return len(self.preds)

    @classmethod
    def from_tensors(cls, classes: Iterable[Tensor], reg: Iterable[Tensor], centerness: Iterable[Tensor]):
        targets = [
            FCOSLevelPrediction(_cls, _reg, _centerness)
            for _cls, _reg, _centerness in zip(classes, reg, centerness)
        ]
        return cls(tuple(targets))


@dataclass(repr=False)
class FCOSLevelTarget(TensorDataclass):
    cls: Tensor
    reg: Tensor
    centerness: Tensor

    cls_ignore: Optional[Tensor]
    reg_ignore: Tensor
    centerness_ignore: Tensor

    stride: int
    interest_range: Tuple[int, int]

    def __iter__(self) -> Iterator[Tensor]:
        yield self.cls
        yield self.reg
        yield self.centerness

    def apply_cls_ignore(self, cls: Tensor) -> Tensor:
        if self.cls_ignore is None:
            return cls
        result = cls[~(self.cls_ignore.expand_as(cls))]
        return result

    def apply_reg_ignore(self, reg: Tensor) -> Tensor:
        B, C, H, W = reg.shape
        result = reg.movedim(1, -1)[~self.reg_ignore]
        return result

    def apply_centerness_ignore(self, centerness: Tensor) -> Tensor:
        B, C, H, W = centerness.shape
        result = centerness.view(B, H, W)[~self.centerness_ignore]
        return result

    @classmethod
    def from_boxes(
        cls, 
        bbox: Tensor, 
        classes: Tensor, 
        num_classes: int,
        stride: int, 
        size_target: Tuple[int, int],
        interest_range: Tuple[int, int],
        center_radius: Optional[int] = None,
        cls_smoothing: float = 0.5,
        cls_ignore: Optional[Tensor] = None,
    ):
        Bb, Nb, Cb = bbox.shape
        Bc, Nc, Cc = classes.shape
        assert Bb == Bc
        assert Nb == Nc
        assert Cb == 4
        assert Cc == 1
        B = Bb
        N = Nb
        assert (classes < num_classes).all(), str(classes)
        H, W = size_target
        if cls_ignore is not None:
            assert isinstance(cls_ignore, Tensor)
            assert cls_ignore.shape in [(B, 1, H, W), (B, Cc, H, W)]

        # mask of boxes that are within the interest range
        bounds = bbox.new_tensor(interest_range).view(1, 2)
        this_level_mask = assign_boxes_to_levels(bbox, bounds=bounds)
        this_level_mask = this_level_mask.view(B, N)

        # ignore boxes that aren't within the desired interest range
        bbox = bbox.clone()
        classes = classes.clone()
        bbox.masked_fill_(~this_level_mask.view(B, N, 1), IGNORE)
        classes.masked_fill_(~this_level_mask.view(B, N, 1), IGNORE)
        padding = (classes == IGNORE)
        assert padding.shape == (B, N, 1)
        assert (bbox[~this_level_mask] == IGNORE).all()
        assert (classes[~this_level_mask] == IGNORE).all()

        # fast shortcut when no boxes
        if padding.all():
            cls_target = torch.zeros(B, num_classes, *size_target, device=classes.device, dtype=torch.float)
            reg_target = bbox.new_empty(B, 4, *size_target).fill_(-1)
            centerness_target = cls_target.new_empty(B, 1, *size_target).fill_(-1)
            reg_ignore = reg_target.new_ones((B, *size_target), dtype=torch.bool)
            centerness_ignore = torch.ones_like(reg_ignore)
            return cls(cls_target, reg_target, centerness_target, cls_ignore, reg_ignore, centerness_ignore, stride, interest_range)

        # get mask of valid locations within each box
        inside_box_mask = bbox_to_mask(bbox, stride, size_target)
        center_mask = bbox_to_mask(bbox, stride, size_target, center_radius=1)
        mask = bbox_to_mask(bbox, stride, size_target, center_radius)
        assert mask.shape == inside_box_mask.shape == (B, N, H, W)

        # build regression target
        reg_target = create_regression_target(bbox, stride, size_target)
        reg_target[~inside_box_mask[..., None, :, :].expand_as(reg_target)] = IGNORE
        assert reg_target.shape == (B, N, 4, H, W)

        # apply mask to regression target and take per pixel minimum for all boxes
        reg_target[reg_target == IGNORE] = INF
        reg_target = reg_target.amin(dim=1)
        reg_target[reg_target == INF] = IGNORE
        assert reg_target.shape == (B, 4, H, W)

        # build centerness target
        centerness_target = compute_centerness_targets(reg_target.movedim(1, -1)).movedim(-1, 1).clamp_min_(0)
        assert centerness_target.shape == (B, 1, H, W)

        # build classification target
        cls_target = create_classification_target(bbox, classes, mask, num_classes)
        cls_target_center = create_classification_target(bbox, classes, center_mask, num_classes)
        assert cls_target.shape == (B, num_classes, H, W)
        if center_radius is None:
            cls_target.mul_(centerness_target).pow_(cls_smoothing)
            cls_target = torch.max(cls_target, cls_target_center)

        reg_ignore = (reg_target == IGNORE).all(dim=1)
        centerness_ignore = inside_box_mask.any(dim=1).logical_not_()
        assert centerness_ignore.shape == (B, H, W)
        assert reg_ignore.shape == (B, H, W)

        return cls(cls_target, reg_target, centerness_target, cls_ignore, reg_ignore, centerness_ignore, stride, interest_range)

    @property
    def background_mask(self) -> Tensor:
        return self.cls <= 0


@dataclass(repr=False)
class FCOSTarget(TensorDataclass):
    targets: Tuple[FCOSLevelTarget, ...]

    def __iter__(self) -> Iterator[FCOSLevelTarget]:
        for t in self.targets:
            yield t

    def __getitem__(self, level: int) -> FCOSLevelTarget:
        return self.targets[level]

    def __len__(self) -> int:
        return len(self.targets)

    @property
    def class_sum(self) -> int:
        return int(sum((t.cls > 0.5).sum() for t in self.targets))

    @property
    def centerness_sum(self) -> int:
        return int(sum((t.centerness > 0).sum() for t in self.targets))

    @property
    def reg_sum(self) -> int:
        return int(sum((t.reg > 0).sum() for t in self.targets))

    @classmethod
    def from_boxes(
        cls, 
        bbox: Tensor, 
        classes: Tensor, 
        num_classes: int,
        strides: Tuple[int, ...], 
        size_targets: Tuple[Tuple[int, int], ...],
        interest_ranges: Tuple[Tuple[int, int], ...],
        center_radius: Optional[int] = None,
        cls_smoothing: float = 0.5,
        cls_ignore: Optional[Tuple[Optional[Tensor], ...]] = None,
    ):
        if cls_ignore is None:
            cls_ignore = (None,) * len(strides) 
        targets: List[FCOSLevelTarget] = []
        for stride, size_target, interest_range, ignore in zip(strides, size_targets, interest_ranges, cls_ignore):
            t = FCOSLevelTarget.from_boxes(
                bbox, 
                classes, 
                num_classes,
                stride,
                size_target,
                interest_range,
                center_radius,
                cls_smoothing,
                ignore,
            )
            targets.append(t)
        return cls(tuple(targets))

    def ignore_background(self, batch_mask: Optional[Tensor] = None) -> "FCOSTarget":
        for t in self.targets:
            mask = t.background_mask
            if batch_mask is not None:
                mask = mask.logical_and(batch_mask.view(-1, 1, 1, 1))
            t.cls_ignore = mask
        return self


@dataclass(repr=False)
class FCOSLossValue(TensorDataclass):
    cls_loss: Tensor
    reg_loss: Tensor
    centerness_loss: Tensor

    def __iter__(self) -> Iterator[Tensor]:
        yield self.cls_loss
        yield self.reg_loss
        yield self.centerness_loss

    @property
    def total_loss(self) -> Tensor:
        return self.cls_loss + 0.1 * self.reg_loss + 0.1 * self.centerness_loss

    @classmethod
    def reduce(
        cls,
        losses: Iterable["FCOSLossValue"],
        target: FCOSTarget,
    ) -> "FCOSLossValue":
        num_gpus = get_num_gpus()

        cls_loss_divisor = max(target.class_sum, 1.0)
        reg_loss_divisor = max(target.reg_sum, 1.0)
        centerness_loss_divisor = max(target.centerness_sum, 1.0)

        cls_loss = sum([t.cls_loss.div(cls_loss_divisor).sum() for t in losses]) 
        reg_loss = sum([t.reg_loss.div(reg_loss_divisor).sum() for t in losses]) 
        centerness_loss = sum([t.centerness_loss.div(centerness_loss_divisor).sum() for t in losses])

        return cls(cls_loss, reg_loss, centerness_loss)


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
        log_reg: bool = False,
        cls_smoothing: float = 1.0,
        focal: bool = True,
    ):
        self.strides = tuple([int(x) for x in strides])
        self.interest_range = tuple([(int(x), int(y)) for x, y in interest_range])
        self.num_classes = int(num_classes)
        self.pad_value = pad_value
        self.radius = int(radius) if radius is not None else None
        self.log_reg = log_reg
        self.cls_smoothing = cls_smoothing

        self.pad_value = float(pad_value)
        self.radius = int(radius) if radius is not None else None

        if focal:
            self.cls_criterion = FocalLossWithLogits(gamma, alpha, reduction="none", label_smoothing=1e-4)
        else:
            self.cls_criterion = nn.BCEWithLogitsLoss(reduction="none")
        #self.reg_criterion = CompleteIoULoss(reduction="none")
        self.reg_criterion = nn.L1Loss(reduction="none")
        self.centerness_criterion = nn.BCEWithLogitsLoss(reduction="none")

    def __call__(
        self,
        cls_pred: Iterable[Tensor],
        reg_pred: Iterable[Tensor],
        centerness_pred: Iterable[Tensor],
        target_bbox: Tensor,
        target_cls: Tensor,
    ) -> FCOSLossValue:
        return self.compute_from_box_target(cls_pred, reg_pred, centerness_pred, target_bbox, target_cls)

    @torch.no_grad()
    def create_targets(
        self, 
        bbox: Tensor, 
        cls: Tensor, 
        size_targets: Tuple[Tuple[int, int], ...],
    ) -> FCOSTarget:
        return FCOSTarget.from_boxes(bbox, cls, self.num_classes, self.strides, size_targets, self.interest_range, self.radius, self.cls_smoothing)

    def compute_from_box_target(
        self,
        cls_pred: Iterable[Tensor],
        reg_pred: Iterable[Tensor],
        centerness_pred: Iterable[Tensor],
        target_bbox: Tensor,
        target_cls: Tensor,
        ignore_background: bool = False,
    ) -> FCOSLossValue:
        size_targets = self.get_size_targets(cls_pred)
        target = FCOSTarget.from_boxes(target_bbox, target_cls, self.num_classes, self.strides, size_targets, self.interest_range, self.radius, self.cls_smoothing)
        if ignore_background:
            target = target.ignore_background()
        pred = FCOSPrediction.from_tensors(cls_pred, reg_pred, centerness_pred)
        return self.compute_from_fcos_target(pred, target)

    @staticmethod
    def get_size_targets(x: Iterable[Tensor]) -> Tuple[Tuple[int, int]]:
        return tuple(tuple(t.shape[-2:]) for t in x) # type: ignore

    def compute_from_fcos_target(
        self,
        pred: FCOSPrediction,
        target: FCOSTarget,
    ) -> FCOSLossValue:
        losses: List[FCOSLossValue] = []
        for _pred, _target in zip(pred, target):
            pred_cls = _target.apply_cls_ignore(_pred.cls)
            target_cls = _target.apply_cls_ignore(_target.cls)
            cls_loss = self.cls_criterion(pred_cls, target_cls)

            pred_reg = _target.apply_reg_ignore(_pred.reg)
            target_reg = _target.apply_reg_ignore(_target.reg)
            if self.log_reg:
                target_reg = target_reg.log().clamp_min(0)
            reg_loss = self.reg_criterion(pred_reg, target_reg)

            pred_centerness = _target.apply_centerness_ignore(_pred.centerness)
            target_centerness = _target.apply_centerness_ignore(_target.centerness)
            centerness_loss = self.centerness_criterion(pred_centerness, target_centerness)

            level_loss = FCOSLossValue(cls_loss, reg_loss, centerness_loss)
            losses.append(level_loss)
        final_loss = FCOSLossValue.reduce(losses, target)
        return final_loss
