#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Tuple

import cv2
import torch
from torch import Tensor

from combustion.util import check_is_tensor


def mask_to_polygon(mask: Tensor, num_classes: int, pad_value: float = -1) -> Tuple[Tuple[Tensor, ...], ...]:
    check_is_tensor(mask, "mask")
    assert mask.ndim == 4
    assert num_classes > 0

    batch_size, _, height, width = mask.shape
    mask = mask.view(batch_size, height, width)

    result = []
    for elem in mask:
        for cls in range(num_classes):
            cls_mask = (elem == cls).byte().numpy()
            contours, hierarchy = cv2.findContours(cls_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            result.append(tuple([torch.from_numpy(x).long() for x in contours]))
    return tuple(result)
