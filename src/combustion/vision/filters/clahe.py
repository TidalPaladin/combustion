#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import torch
from torch import Tensor


class CLAHE:
    r"""Contrast Limited Adaptive Histogram Equalization. Uses CV2 as the the
    computational backend. Input should have either a single luminance channel or three RGB channels.

    Args:

        ``*args``:
            Passed to :class:`cv2.createCLAHE`.

        ``*kwargs``:
            Passed to :class:`cv2.createCLAHE`.

    Shape:
        * ``inputs`` - :math:`(N, C, H, W)` where :math:`C \in \{1, 3\}`
        * Outputs - same as inputs

    """

    def __init__(self, *args, **kwargs):
        self.clahe = cv2.createCLAHE(*args, **kwargs)

    def __call__(self, inputs: Tensor) -> Tensor:
        if inputs.dtype != torch.uint8:
            raise TypeError(f"CLAHE only supports torch.uint8, found {inputs.dtype}")
        if inputs.ndim != 4:
            raise ValueError(f"Expected inputs.ndim == 4, found {inputs.ndim}")

        inputs.device
        batch_size, channels, height, width = inputs.shape

        result = torch.empty_like(inputs)
        for i in range(batch_size):
            batch_elem = inputs[i]

            if channels == 1:
                batch_elem = batch_elem.view(height, width)
                output = torch.from_numpy(self.clahe.apply(batch_elem.numpy()))
            else:
                # RGB -> LAB and extract luminance for CLAHE
                batch_elem = batch_elem.permute(1, 2, 0).numpy()
                batch_elem = cv2.cvtColor(batch_elem, cv2.COLOR_RGB2LAB)
                luminance = batch_elem[0, ...]

                # apply CLAHE, then back to RGB
                luminance = self.clahe.apply(luminance)
                batch_elem[0, ...] = luminance
                output = torch.from_numpy(cv2.cvtColor(batch_elem, cv2.COLOR_LAB2RGB)).permute(-1, 0, 1)

            result[i] = output.type_as(result).view(channels, height, width)

        return result
