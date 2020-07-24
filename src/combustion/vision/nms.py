#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Tuple

import torch
from torch import Tensor


try:
    from torchvision.ops import nms as nms_torch
except ImportError:

    def nms_torch(*args, **kwargs):
        raise ImportError(
            "nms requires kornia. "
            "Please install combustion with 'vision' extras using "
            "pip install combustion [vision]"
        )


def nms(boxes: Tensor, scores: Tensor, iou_threshold: float) -> Tuple[Tensor, Tuple]:
    r"""Performs non-maximal suppression on anchor boxes as per `torchvision.ops.nms`.
    Supports batched or non-batched inputs, and returns a tuple of index tensors that
    can be used to index the input boxes / scores tensors.

    Args:
        boxes (tensor):
            The anchor boxes to perform non-maximal suppression on.

        scores (tensor):
            The confidence scores associated with each tensor.

        iou_threshold (float):
            Value on the interval :math:`[0, 1]` giving the intersection over union
            threshold over which non-maximal boxes will be suppressed.

    Shape:
        - Boxes: :math:`(N, 4)` or :math:`(B, N, 4)` where :math:`B` is an optional batch
          dimension and `N` is the number of anchor boxes.
        - Scores: :math:`(N)` or :math:`(B, N)` where :math:`B` is an optional batch
          dimension and `N` is the number of anchor boxes.
        - Output: Tensor tuple giving the maximal indices, each of shape :math:`(K)`.

    Example::

        >>> boxes = torch.tensor([[
        >>>     [0., 0., 10., 10.],
        >>>     [1., 1., 11., 11.],
        >>>     [10., 10., 20., 20.]
        >>> ]])

        >>> scores = torch.tensor([[0.1, 0.5, 0.05]])

        >>> nms_indices = nms(boxes, scores, threshold=0.5)
        >>> nms_boxes, nms_scores = boxes[nms_indices], scores[nms_indices]
    """
    # batched recursion
    if boxes.ndim == 3:
        batch_size, num_boxes = boxes.shape[0:2]
        outputs = []
        for i, example in enumerate(zip(boxes, scores)):
            nms_indices = nms_torch(*example, iou_threshold)
            batch_idx = torch.empty_like(nms_indices).fill_(i)
            outputs.append((batch_idx, nms_indices))

        batch_indices = torch.cat(list(zip(*outputs))[0], 0)
        box_indices = torch.cat(list(zip(*outputs))[1], 0)
        return batch_indices, box_indices

    else:
        return nms_torch(boxes, scores, iou_threshold)
