#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import torch
from torch import Tensor


class CLAHE:
    r"""Contrast Limited Adaptive Histogram Equalization. Uses CV2 as the the
    computational backend. Input should have either a single luminance channel or three RGB channels.
    Expected datatypes for ``input`` are ``torch.uint8``, ``torch.uint16``, or any floating point
    datatype with values in the range ``[0, 1]``.

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
        self._args = args
        self._kwargs = kwargs

    def __repr__(self):
        s = "CLAHE("
        for arg in self._args:
            s += f"{arg}, "
        for key, value in self._kwargs.items():
            s += f"{key}={value}, "
        s = s[:-2] + ")"
        return s

    def __call__(self, inputs: Tensor) -> Tensor:
        if inputs.dtype != torch.uint8 and not inputs.is_floating_point():
            raise TypeError(f"CLAHE only supports torch.uint8 or float in range [0, 1], found {inputs.dtype}")
        if inputs.ndim != 4:
            raise ValueError(f"Expected inputs.ndim == 4, found {inputs.ndim}")

        batch_size, channels, height, width = inputs.shape
        result = torch.empty_like(inputs)
        inputs = inputs.cpu()

        # cv2 clahe doesn't support float, so if needed we will convert to uint16
        is_float = inputs.is_floating_point()

        for i in range(batch_size):
            batch_elem = inputs[i]

            # grayscale CLAHE
            if channels == 1:
                batch_elem = batch_elem.view(height, width)
                if is_float:
                    batch_elem = batch_elem.mul(2 ** 16).numpy().astype(np.uint16)
                assert batch_elem.dtype in [np.uint8, np.uint16]
                output = self.clahe.apply(batch_elem)
                if is_float:
                    output = output.astype(float)
                output = torch.from_numpy(output)

            # color CLAHE
            else:
                # RGB -> YUV and extract luminance for CLAHE
                batch_elem = batch_elem.permute(1, 2, 0)
                if is_float:
                    batch_elem = batch_elem.mul(2 ** 16).numpy().astype(np.uint16)
                batch_elem = cv2.cvtColor(batch_elem, cv2.COLOR_RGB2YUV)
                luminance = batch_elem[0, ...]

                # apply CLAHE to luminance channel and update
                assert batch_elem.dtype in [np.uint8, np.uint16]
                luminance = self.clahe.apply(luminance)
                batch_elem[0, ...] = luminance

                # convert CLAHE YUV channel back to RGB
                output = cv2.cvtColor(batch_elem, cv2.COLOR_YUV2RGB)
                if is_float:
                    output = output.astype(float)

                # restore numpy array to channels first tensor
                output = torch.from_numpy(output).permute(-1, 0, 1)

            # restore floats to range [0, 1]
            if is_float:
                output = output.type_as(result).div_(2 ** 16)

            result[i] = output.type_as(result).view(channels, height, width)

        return result
