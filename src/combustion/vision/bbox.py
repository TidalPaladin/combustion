#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Dict, Optional, Tuple, Union

import torch
from numpy import ndarray
from torch import Tensor


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
    bbox: Optional[Union[Tensor, ndarray]] = None,
    label: Optional[Union[Tensor, ndarray]] = None,
    scores: Optional[Union[Tensor, ndarray]] = None,
    class_names: Optional[Dict[int, str]] = None,
    box_color: Tuple[int, int, int] = (255, 0, 0),
    text_color: Tuple[int, int, int] = (255, 255, 255),
    label_alpha: float = 0.4,
    thickness: int = 2,
) -> Union[Tensor, ndarray]:
    r"""Adds bounding box visualization to an input array

    Args:
        img (Tensor or numpy.ndarray):
            The image to draw anchor boxes on.

        bbox (Tensor or numpy.ndarray, optional):
            The anchor boxes to draw

        label (Tensor or numpy.ndarray, optional):
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

    Returns:
        :class:`torch.Tensor` or :class:`numpy.ndarray` (depending on what was given for `img`)
        with the output image.
    """
    original_img_type = type(img)

    img: Tensor = _check_input(img, "img", (2, 3))
    bbox: Optional[Tensor] = _check_input(bbox, "bbox", 2, (None, 4))
    label: Optional[Tensor] = _check_input(label, "label", 2, (None, 1))
    scores: Optional[Tensor] = _check_input(scores, "scores", 2, (None, 1))

    # permute to channels last
    if img.ndim == 3:
        img = img.permute(1, 2, 0)
    else:
        img = img.unsqueeze(-1)

    img, bbox, label = [x.cpu().numpy() if x is not None else None for x in (img, bbox, label)]

    # convert grayscale input to color for bounding boxes
    if img.shape[-1] < 3:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    if bbox is not None:
        for i, coords in enumerate(bbox):
            x_min, y_min, x_max, y_max = [int(c) for c in coords]

            # bounding box
            cv2.rectangle(
                img, (x_min, y_min), (x_max, y_max), box_color, thickness,
            )

            if label is not None:
                cls = label[i].item()
                if class_names is not None:
                    class_name = class_names[cls]
                else:
                    class_name = f"Class {cls}"

                if scores is not None:
                    class_name += f" - {scores[i].item():0.3f}"

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

    # restore original data type with channels first ordering
    channel, height, width = -1, 0, 1
    if original_img_type == Tensor:
        img = torch.from_numpy(img)
        img = img.permute(channel, height, width)
    else:
        img = img.transpose(channel, height, width)

    return img


def _check_input(x, name, ndim=None, shape=None):
    if name != "img" and x is None:
        return None
    elif x is None:
        raise ValueError("img cannot be None")

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
