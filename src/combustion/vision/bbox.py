#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Dict, Optional, Tuple, Union

import torch
from numpy import ndarray
from torch import Tensor


try:
    import cv2
except ImportError:
    print("combustion.visualization requires the cv2 module")
    raise

CORRECT_BOX_COLOR = BOX_COLOR = (255, 0, 0)
TEXT_COLOR = (255, 255, 255)


def visualize_bbox(
    img: Union[Tensor, ndarray],
    bbox: Union[Tensor, ndarray],
    label: Union[Tensor, ndarray],
    class_names: Optional[Dict[int, str]] = None,
    box_color: Tuple[int, int, int] = (255, 0, 0),
    text_color: Tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2,
) -> Union[Tensor, ndarray]:
    r"""Adds bounding box visualization to an input array
    """
    original_img_type = type(img)

    img: Tensor = _check_input(img, "img", (2, 3))
    bbox: Tensor = _check_input(bbox, "bbox", 2, (None, 4))
    label: Tensor = _check_input(label, "label", 2, (None, 1))

    # permute to channels last
    if img.ndim == 3:
        img = img.permute(1, 2, 0)
    else:
        img = img.unsqueeze(-1)

    img, bbox, label = [x.cpu().numpy() for x in (img, bbox, label)]

    # convert grayscale input to color for bounding boxes
    if img.shape[-1] < 3:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # add bbox and class label for each box
    for coords, cls in zip(bbox, label):
        x_min, y_min, x_max, y_max = [int(c) for c in coords]

        # bounding box
        cv2.rectangle(
            img, (x_min, y_min), (x_max, y_max), box_color, thickness,
        )

        if class_names is not None:
            class_name = class_names[cls]
        else:
            class_name = f"Class {cls}"

        # tag bounding box with class name / integer id
        ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), box_color, -1)
        cv2.putText(
            img,
            class_name,
            (x_min, y_min - int(0.3 * text_height)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            text_color,
            lineType=cv2.LINE_AA,
        )

    if original_img_type == Tensor:
        img = torch.from_numpy(img)

    return img


def _check_input(x, name, ndim=None, shape=None):
    if not isinstance(x, (Tensor, ndarray)):
        raise TypeError(f"Expected Tensor or np.ndarray for {name}, found {type(x)}")

    if isinstance(x, ndarray):
        x = torch.from_numpy(x)

    if ndim is not None:
        if isinstance(ndim, int):
            if x.ndim != ndim:
                raise ValueError(f"Expected {name}.ndim = {ndim}, found {x.ndim}")
        elif x.ndim < ndim[0] or x.ndim > ndim[1]:
            raise ValueError(f"Expected {ndim[0]} <= {name}.ndim = {ndim[1]}, found {x.ndim}")

    if shape is not None:
        for i, dim in enumerate(shape):
            if dim is not None and x.shape[i] != dim:
                raise ValueError(f"Expected {name}.shape = {shape}, found {x.shape}")

    return x
