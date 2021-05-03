#!/usr/bin/env python
# -*- coding: utf-8 -*-


from abc import ABC, abstractstaticmethod
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.ops import batched_nms

from combustion.vision import batch_box_target

from .fpn_shared_head import SharedDecoder2d


class BaseFCOSDecoder(nn.Module, ABC):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        num_regressions: int,
        num_convs: int,
        strides: Optional[Tuple[int]] = None,
        activation: nn.Module = nn.SiLU(),
        reg_activation: nn.Module = nn.ReLU(),
        num_groups: int = 32,
        gn_epsilon: float = 1e-5,
        cls_prior: float = 0.01,
    ):
        super().__init__()
        self.cls_head = SharedDecoder2d(
            in_channels,
            num_classes,
            num_convs,
            scaled=False,
            strides=strides,
            activation=activation,
            final_activation=nn.Identity(),
            num_groups=num_groups,
            gn_epsilon=gn_epsilon,
        )

        self.reg_head = SharedDecoder2d(
            in_channels,
            num_regressions + 1,
            num_convs,
            scaled=False,
            strides=strides,
            activation=activation,
            final_activation=nn.Identity(),
            num_groups=num_groups,
            gn_epsilon=gn_epsilon,
        )
        self.reg_activation = reg_activation
        self.strides = strides

        bias_value = torch.tensor(cls_prior).logit()
        torch.nn.init.constant_(self.cls_head.final_conv_pw.bias, bias_value)

    def forward(self, fpn: Tuple[Tensor]) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        cls = self.cls_head(fpn)
        _ = self.reg_head(fpn)
        centerness = [layer[..., 0:1, :, :] for layer in _]
        reg = [self.reg_activation(layer[..., 1:, :, :] * s) for s, layer in zip(self.strides, _)]
        return cls, reg, centerness

    @abstractstaticmethod
    def postprocess(cls: List[Tensor], reg: List[Tensor], centerness: List[Tensor]) -> Tensor:
        raise NotImplementedError()

    @staticmethod
    def reduce_heatmaps(
        heatmap: Tuple[Tensor, ...],
        reduction: Callable[[Tensor, Tensor], Tensor] = torch.max,
        channel_reduction: Callable[[Tensor, int, bool], Tensor] = torch.amax,
        mode: str = "nearest",
    ) -> Tensor:
        r"""Helper function that reduces FCOS FPN heatmaps into a single channel heatmap
        suitable for visualization.

        Args:
            heatmap (tuple of :class:`torch.Tensor`):
                FCOS FPN heatmaps to reduce

            reduction:
                Function that should accept two equally sized tensors and reduce them to a
                single output tensor. By default, heatmaps are reduced with :func:`torch.max`.

            channel_reduction:
                Function that should accept a tensor with multiple channels and reduce it to
                a single channel output tensor. By default, channels are reduced with
                :func:`torch.amax` along ``dim=1``.
        """
        result = heatmap[0]
        C = result.shape[1]

        # reduce each FPN level
        for i in range(len(heatmap) - 1):
            current_level = F.interpolate(heatmap[i + 1], result.shape[-2:], mode=mode)
            result = reduction(current_level, result)

        # reduce across channels to a 1 channel heatmap
        if C != 1:
            result = torch.amax(result, dim=1, keepdim=True)

        return result


def _postprocess_level(
    stride: int, level_cls: Tensor, level_reg: Tensor, level_centerness: Tensor, threshold: float, from_logits: bool
):
    if from_logits:
        level_cls = torch.sigmoid(level_cls)
        level_centerness = torch.sigmoid(level_centerness)

    # scale classifications based on centerness
    scaled_score = (level_cls * level_centerness.expand_as(level_cls)).sqrt_()

    # get indices of positions that exceed threshold
    positive_locations = (level_cls >= threshold).nonzero()

    # extract coordinates of positive predictions and drop scores for negative predictions
    batch, class_id, y, x = positive_locations.split(1, dim=-1)
    raw_score = level_cls[batch, class_id, y, x]
    scaled_score = scaled_score[batch, class_id, y, x]

    assert not raw_score.isnan().any()
    assert not scaled_score.isnan().any()

    # use stride to compute base coodinates within the original image
    # use pred regression to compute l, t, r, b offset
    base = positive_locations[..., -2:].roll(1, -1).float().mul_(stride).add_(stride / 2.0).repeat(1, 2)
    offset = level_reg[batch, :, y, x].view_as(base)
    offset[..., :2].neg_()

    # compute final regressions and clamp to lie within image_size
    coords = base + offset

    # record the boxes and box -> batch mapping
    boxes = torch.cat([coords, raw_score, scaled_score, class_id], dim=-1)
    return boxes, batch


def _apply_nms(
    final_boxes: Tensor, final_batch_idx: Tensor, nms_threshold: float, num_classes: int
) -> Tuple[Tensor, Tensor]:
    coords = final_boxes[..., :4]
    final_boxes[..., -3, None]
    scaled_score = final_boxes[..., -2, None]
    class_id = final_boxes[..., -1, None]

    # torchvision NMS cant do batches of images, but it can separate based on class id
    # create a new "class id" that distinguishes batch and class
    idx = (final_batch_idx * num_classes + class_id.view_as(final_batch_idx)).view(-1).long()
    keep = batched_nms(coords.float(), scaled_score.view(-1), idx, nms_threshold)
    final_boxes = final_boxes[keep, :]
    final_batch_idx = final_batch_idx[keep, :]
    return final_boxes.contiguous(), final_batch_idx.contiguous()


def _apply_pre_nms_limit(
    final_boxes: Tensor, final_batch_idx: Tensor, limit: int, batch_size: int
) -> Tuple[Tensor, Tensor]:
    # restrict top k boxes prior to NMS to avoid memory explosion
    pre_nms_top_k: List[Tensor] = []
    pre_nms_top_k_batch: List[Tensor] = []
    for i in range(batch_size):
        # find indices of top k highest scaled score boxes within the batch
        topk = final_boxes[(final_batch_idx == i).view(-1), -2].argsort(descending=True)[:limit]

        # use indices to extract boxes from this batch and update final_batch_idx
        values = final_boxes[(final_batch_idx == i).view(-1)][topk, :]
        pre_nms_top_k.append(values)
        values = final_batch_idx[(final_batch_idx == i).view(-1)][topk, :]
        pre_nms_top_k_batch.append(values)

    final_boxes = torch.cat(pre_nms_top_k)
    final_batch_idx = torch.cat(pre_nms_top_k_batch)
    return final_boxes, final_batch_idx


class FCOSDecoder(BaseFCOSDecoder):
    r"""Decoder for Fully Convolutional One-Stage Object Detector (FCOS) as described
    in PAPER. FCOS is an anchor-free object detection implementation that predicts
    detection, regression, and centerness heatmaps at each FPN level. These predictions
    are postprocessed to create a set of anchor boxes.

    Args:
        in_channels (int):
            Number of input channels at each FPN level.

        num_classes (int):
            Number of classes to detect.

        num_convs (int):
            Number of convolutional repeats in each decoder head.

        strides (tuple of ints, optional):
            Strides at each FPN level.  By default, assume each FPN level differs
            in stride by a factor of 2.

        activation (nn.Module):
            Activation function for each intermediate repeat in the heads.

        bn_momentum (float):
            Momentum value for batch norm

        bn_epsilon (float):
            Epsilon value for batch norm

    Returns:
        List of classification, regression, and centerness predictions for each
        FPN level.

    Shape:
        * ``fpn`` - :math:`(N, C, H_i, W_i)` where :math:`i` is the :math:`i`'th FPN level
        * Classification - :math:`(N, O, H_i, W_i)` where :math:`O` is the number of classes
        * Regression - :math:`(N, 4, H_i, W_i)`
        * Centerness - :math:`(N, 1, H_i, W_i)`
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        num_convs: int,
        strides: Optional[Tuple[int]] = None,
        activation: nn.Module = nn.SiLU(),
        num_groups: float = 32,
        gn_epsilon: float = 1e-5,
    ):

        super().__init__(
            in_channels,
            num_classes,
            4,
            num_convs,
            strides,
            activation,
            nn.ReLU(),
            num_groups,
            gn_epsilon,
        )

    @staticmethod
    def postprocess(
        cls: List[Tensor],
        reg: List[Tensor],
        centerness: List[Tensor],
        strides: List[int],
        threshold: float = 0.05,
        pad_value: float = -1,
        from_logits: bool = False,
        nms_threshold: Optional[float] = 0.5,
        use_raw_score: bool = False,
        max_boxes: Optional[int] = None,
        pre_nms_max_boxes: Optional[int] = 1000,
    ) -> Tensor:
        r"""Postprocesses detection, regression, and centerness predictions into a set
        of anchor boxes.

        Args:
            cls (iterable of tensors):
                Classification predictions at each FPN level

            reg (iterable of tensors):
                Regression predictions at each FPN level

            centerness (iterable of tensors):
                Centerness predictions at each FPN level

            strides (tuple of ints):
                Strides at each FPN level.

            threshold (float):
                Detection confidence threshold

            from_logits (bool):
                If ``True``, assume that ``cls`` and ``centerness`` are logits and not
                probabilities.

            nms_threshold (float, optional):
                Threshold for non-maximal suppression. If ``None``, do not apply NMS.

            use_raw_score (bool):
                If ``True``, assign scores to boxes based on their predicted classification score.
                Otherwise, scores are assigned based on classification and centerness scores.

            max_boxes (int, optional):
                An optional limit on the maximum number of boxes per image

            pre_nms_max_boxes (int, optional):
                An optional limit on the maximum number of boxes per image before NMS

        Returns:
            Predicted boxes in the form :math:`(x_1, y_1, x_2, y_x, score, class)`.

        Shape:
            * ``cls`` - :math:`(*, C, H_i, W_i)` where :math:`i` is the :math:`i`'th FPN level
            * ``reg`` - :math:`(*, 4, H_i, W_i)` where :math:`i` is the :math:`i`'th FPN level
            * ``centerness`` - :math:`(*, 1, H_i, W_i)` where :math:`i` is the :math:`i`'th FPN level
            * Output - :math:`(*, N, 6)`
        """
        torch.autograd.set_grad_enabled(False)
        threshold = abs(float(threshold))
        nms_threshold = abs(float(nms_threshold)) if nms_threshold is not None else None
        assert len(strides) == len(cls)
        assert len(strides) == len(reg)
        assert len(strides) == len(centerness)

        _ = [x * strides[0] for x in cls[0].shape[-2:]]
        y_lim, x_lim = _
        assert x_lim > 0
        assert y_lim > 0

        batch_idx, boxes = [], []

        batch_size = cls[0].shape[0]
        num_classes = cls[0].shape[1]

        # iterate over each FPN level
        for i, (stride, level_cls, level_reg, level_centerness) in enumerate(zip(strides, cls, reg, centerness)):
            bbox, batch = _postprocess_level(stride, level_cls, level_reg, level_centerness, threshold, from_logits)
            boxes.append(bbox)
            batch_idx.append(batch)

        # combine boxes across all FPN levels
        if len(boxes):
            final_boxes = torch.cat(boxes, dim=-2)
            final_batch_idx = torch.cat(batch_idx, dim=-2)
            del boxes
            del batch_idx
        # handle case of no boxes across entire batch
        else:
            return reg[0].new_empty(batch_size, 0, 6)

        # restrict to top k boxes before NMS to avoid exploding memory
        if pre_nms_max_boxes is not None:
            final_boxes, final_batch_idx = _apply_pre_nms_limit(
                final_boxes, final_batch_idx, pre_nms_max_boxes, batch_size
            )

        # ensure boxes are bounded within image area
        coords = final_boxes[..., :4]
        coords.clamp_min_(0)
        coords[..., 2].clamp_max_(x_lim)
        coords[..., 3].clamp_max_(y_lim)

        # apply NMS to final_boxes
        if nms_threshold is not None:
            final_boxes, final_batch_idx = _apply_nms(final_boxes, final_batch_idx, nms_threshold, num_classes)

        # create final box using raw or centerness adjusted score as specified
        coords, raw_score, scaled_score, class_id = final_boxes.split((4, 1, 1, 1), dim=-1)
        if use_raw_score:
            final_boxes = torch.cat((coords, raw_score, class_id), dim=-1)
        else:
            final_boxes = torch.cat((coords, scaled_score, class_id), dim=-1)

        # ensure boxes sorted by score
        if use_raw_score or nms_threshold is None:
            score = final_boxes[..., -2]
            sorted_idx = score.argsort(dim=-1, descending=True)
            final_boxes = final_boxes[sorted_idx, :]
            final_batch_idx = final_batch_idx[sorted_idx]

        # pack final_boxes into a padded batch
        final_boxes = [final_boxes[(final_batch_idx == i).view(-1), :] for i in range(batch_size)]
        final_boxes = batch_box_target(final_boxes, pad_value)

        # enforce max box limit if one is given
        if max_boxes is not None:
            final_boxes = final_boxes[..., :max_boxes, :]

        torch.autograd.set_grad_enabled(True)
        return final_boxes
