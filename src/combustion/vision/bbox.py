#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Dict, Optional, Tuple, Union

import torch
from numpy import ndarray
from torch import Tensor

from combustion.util import check_dimension, check_dimension_match, check_is_array, check_ndim_match

from .convert import to_8bit


try:
    import cv2
except ImportError:

    class cv2:
        def __getattr__(self, attr):
            raise ImportError(
                "Bounding box visualization requires cv2. "
                "Please install combustion with 'vision' extras using "
                "pip install combustion [vision]"
            )

        def __setattr__(self, attr):
            raise ImportError(
                "Bounding box visualization requires cv2. "
                "Please install combustion with 'vision' extras using "
                "pip install combustion [vision]"
            )


CORRECT_BOX_COLOR = BOX_COLOR = (255, 0, 0)
TEXT_COLOR = (255, 255, 255)


def visualize_bbox(
    img: Union[Tensor, ndarray],
    bbox: Union[Tensor, ndarray],
    classes: Optional[Union[Tensor, ndarray]] = None,
    scores: Optional[Union[Tensor, ndarray]] = None,
    class_names: Optional[Dict[int, str]] = None,
    box_color: Tuple[int, int, int] = (255, 0, 0),
    text_color: Tuple[int, int, int] = (255, 255, 255),
    label_alpha: float = 0.4,
    thickness: int = 2,
    pad_value: float = -1,
) -> Tensor:
    r"""Adds bounding box visualization to an input array

    Args:
        img (Tensor or numpy.ndarray):
            Background image

        bbox (Tensor or numpy.ndarray):
            Anchor boxes to draw

        classes (Tensor or numpy.ndarray, optional):
            Class labels associated with each anchor box

        scores (Tensor or numpy.ndarray, optional):
            Class scores associated with each anchor box

        class_names (dict, optional):
            Dictionary mapping integer class labels to string names.
            If ``label`` is supplied but ``class_names`` is not, integer
            class labels will be used.

        box_color (tuple of ints, optional):
            A 3-tuple giving the RGB color value to use for anchor boxes.

        text_color (tuple of ints, optional):
            A 3-tuple giving the RGB color value to use for labels.

        label_alpha (float, optional):
            Alpha to apply to the colored background for class labels.

        thickness (int, optional):
            Specifies the thickness of anchor boxes.

        pad_value (float, optional):
            The padding value used when batching boxes and labels

    Returns:
        :class:`torch.Tensor` or :class:`numpy.ndarray` (depending on what was given for `img`)
        with the output image.

    Shape:
        * ``img`` - :math:`(B, C, H, W)` or :math:`(C, H, W)` or :math:`(H, W)`
        * ``bbox`` - :math:`(B, N, 4)` or :math:`(N, 4)`
        * ``classes`` - :math:`(B, N, 1)` or :math:`(N, 1)`
        * ``scores`` - :math:`(B, N, S)` or :math:`(N, S)`
        *  Output - same as ``img``
    """
    # type check
    check_is_array(img, "img")
    check_is_array(bbox, "bbox")
    classes is None or check_is_array(classes, "classes")
    scores is None or check_is_array(scores, "scores")

    # ndim check
    classes is None or check_ndim_match(bbox, classes, "bbox", "classes")
    scores is None or check_ndim_match(bbox, scores, "bbox", "scores")

    # more ndim checks, ensure if one input is batched then all inputs are batched
    boxes_batched = bbox.ndim == 3
    img_batched = img.ndim == 4
    if img_batched != boxes_batched:
        raise ValueError(f"Expected bbox.ndim == 3 when img.ndim == 4, found {bbox.shape}, {img.shape}")
    if boxes_batched:
        if classes is not None and classes.ndim != 3:
            raise ValueError(f"Expected classes.ndim == 3, found {classes.ndim}")
        if scores is not None and scores.ndim != 3:
            raise ValueError(f"Expected scores.ndim == 3, found {scores.ndim}")
    batched = img_batched

    # individual dimension checks
    check_dimension(bbox, dim=-1, size=4, name="bbox")
    classes is None or check_dimension(classes, dim=-1, size=1, name="classes")
    classes is None or check_dimension_match(bbox, classes, -2, "bbox", "classes")
    scores is None or check_dimension_match(bbox, scores, -2, "bbox", "scores")
    img_shape = img.shape[-2:]

    # convert to cpu tensor
    img, bbox, classes, scores = [
        torch.as_tensor(x).cpu() if x is not None else x for x in (img, bbox, classes, scores)
    ]

    # add a channel dimension to img if not present
    if img.ndim == 2:
        img = img.view(1, *img.shape)

    # add a batch dimension if not present
    img = img.view(1, *img.shape) if not batched else img
    bbox = bbox.view(1, *bbox.shape) if not batched else bbox
    if classes is not None:
        classes = classes.unsqueeze(0) if not batched else classes
    if scores is not None:
        scores = scores.unsqueeze(0) if not batched else scores

    # convert image to 8-bit and convert to channels_last
    img = to_8bit(img.clone(), per_channel=False, same_on_batch=True)
    img = img.permute(0, 2, 3, 1).contiguous()

    # convert img to color if grayscale input
    if img.shape[-1] == 1:
        img = img.repeat(1, 1, 1, 3)

    # get box indices that arent padding
    valid_indices = (bbox == pad_value).all(dim=-1).logical_not_()

    # iterate over each batch, building bbox overlay
    result = []
    batch_size = bbox.shape[0]
    for batch_idx in range(batch_size):
        # if this fails with cryptic cv errors, ensure that img is contiguous
        result_i = img[batch_idx].numpy()

        # extract valid boxes for this batch
        valid_indices_i = valid_indices[batch_idx]
        bbox_i = bbox[batch_idx][valid_indices_i]
        scores_i = scores[batch_idx][valid_indices_i] if scores is not None else None
        classes_i = classes[batch_idx][valid_indices_i] if classes is not None else None

        # loop over each box and draw the annotation onto result_i
        for box_idx, coords in enumerate(bbox_i):
            x_min, y_min, x_max, y_max = [int(c) for c in coords]

            # draw the bounding box
            cv2.rectangle(
                result_i,
                (x_min, y_min),
                (x_max, y_max),
                box_color,
                thickness,
            )

            # add class labels to bounding box text if present
            text = ""
            if classes_i is not None:
                cls = int(classes_i[box_idx].item())
                # use class integer -> str name if mapping is given, otherwise use class integer
                if class_names is not None:
                    text += class_names[cls]
                else:
                    text += f"Class {cls}"

            # add score labels to bounding box text if present
            if scores_i is not None:
                if classes_i is not None:
                    text += " - "
                # add the first score
                text += f"{scores_i[box_idx, 0].item():0.3f}"
                # if multiple scores are present, add those
                num_scores = scores_i.shape[-1]
                for score_idx in range(1, num_scores):
                    text += f" | {scores_i[box_idx, score_idx].item():0.3f}"

            # tag bounding box with class name / integer id
            ((text_width, text_height), _) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
            cv2.rectangle(result_i, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), box_color, -1)
            cv2.putText(
                result_i,
                text,
                (x_min, y_min - int(0.3 * text_height)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                text_color,
                lineType=cv2.LINE_AA,
            )

        # permute back to channels first and add to result list
        result_i = torch.from_numpy(result_i).permute(-1, 0, 1)
        result.append(result_i)

    if len(result) > 1:
        result = torch.stack(result, dim=0)
    else:
        result = result[0]

    # ensure we include a batch dim if one was present in inputs
    if batched and batch_size == 1:
        result = result.view(1, *result.shape)

    # convert result to 8-bit
    if result.dtype != torch.uint8:
        result = to_8bit(result, per_channel=False, same_on_batch=True)

    return result
