#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List, Tuple, Union

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# from combustion.nn import HardSigmoid
from combustion.util import double, single, triple


class _RASPPMeta(type):
    def __new__(cls, name, bases, dct):
        x = super().__new__(cls, name, bases, dct)
        if "3d" in name:
            x.Conv = nn.Conv3d
            x.BatchNorm = nn.BatchNorm3d
            x.AvgPool = nn.AvgPool3d
            x.Tuple = staticmethod(triple)
        elif "2d" in name:
            x.Conv = nn.Conv2d
            x.BatchNorm = nn.BatchNorm2d
            x.AvgPool = nn.AvgPool2d
            x.Tuple = staticmethod(double)
        elif "1d" in name:
            x.Conv = nn.Conv1d
            x.BatchNorm = nn.BatchNorm1d
            x.AvgPool = nn.AvgPool1d
            x.Tuple = staticmethod(single)
        else:
            raise RuntimeError(f"Metaclass: error processing name {cls.__name__}")
        return x


class _RASPPLite(nn.Module):
    def __init__(
        self,
        input_filters: int,
        residual_filters: int,
        output_filters: int,
        num_classes: int,
        pool_kernel: Union[int, Tuple[int, ...]] = 42,
        pool_stride: Union[int, Tuple[int, ...]] = 18,
        dilation: Union[int, Tuple[int, ...]] = 1,
        sigmoid: nn.Module = nn.Sigmoid(),
        relu: nn.Module = nn.ReLU(),
        bn_momentum: float = 0.1,
        bn_epsilon: float = 1e-5,
    ):
        super().__init__()
        pool_kernel = self.Tuple(pool_kernel)
        pool_stride = self.Tuple(pool_stride)
        dilation = self.Tuple(dilation)

        self.pooled = nn.Sequential(
            self.AvgPool(pool_kernel, stride=pool_stride),
            self.Conv(input_filters, output_filters, kernel_size=1, stride=pool_stride),
            sigmoid,
        )

        self.main_conv1 = nn.Sequential(
            self.Conv(input_filters, output_filters, kernel_size=1, bias=False),
            self.BatchNorm(output_filters, momentum=bn_momentum, eps=bn_epsilon),
            relu,
        )

        self.residual_conv = self.Conv(residual_filters, num_classes, kernel_size=1)
        self.main_conv2 = self.Conv(output_filters, num_classes, kernel_size=1)

    def forward(self, inputs: List[Tensor]) -> Tensor:
        skip_path, main_path = inputs
        residual = self.residual_conv(skip_path)

        pooled = self.pooled(main_path)
        main = self.main_conv1(main_path)

        # upsample for main to match pooled
        upsample_shape: List[int] = []
        for i, size in enumerate(main.shape):
            if i >= 2:
                upsample_shape.append(size)
        pooled = F.interpolate(pooled, size=upsample_shape, mode="bilinear")

        main = main * pooled

        # upsample for main to match residual
        upsample_shape: List[int] = []
        for i, size in enumerate(residual.shape):
            if i >= 2:
                upsample_shape.append(size)
        main = F.interpolate(main, size=upsample_shape, mode="bilinear")

        main = self.main_conv2(main)
        output = main + residual
        return output


class RASPPLite2d(_RASPPLite, metaclass=_RASPPMeta):
    r"""Implements the a lite version of the reduced atrous spatial pyramid pooling
    module (R-ASPP Lite) described in `Searching for MobileNetV3`_. This is a semantic
    segmentation head.

    .. image:: ./raspp.png
        :width: 800px
        :align: center
        :height: 300px
        :alt: Diagram of R-ASPP Lite.

    Args:
        input_filters (int):
            Number of input channels along the main pathway

        residual_filters (int):
            Number of input channels along the residual pathway

        output_filters (int):
            Number of channels in the middle of the segmentation head.

        num_classes (int):
            Number of classes for semantic segmentation

        pool_kernel (int or tuple of ints):
            Size of the average pooling kernel

        pool_stride (int or tuple of ints):
            Stride of the average pooling kernel

        dilation (int or tuple of ints):
            Dilation of the atrous convolution. Defaults to ``1``, meaning no
            atrous convolution.

        sigmoid (:class:`torch.nn.Module`):
            Activation function to use along the pooled pathway

        relu (:class:`torch.nn.Module`):
            Activation function to use along the main convolutional pathway

        bn_momentum (float):
            Batch norm momentum

        bn_epsilon (float):
            Batch norm epsilon

        final_upsample (int):
           An optional amount of additional to be applied via transposed convolutions.
           It is expected that additional upsampling is a power of two.

    .. _Searching for MobileNetV3:
        https://arxiv.org/abs/1905.02244
    """


class RASPPLite1d(_RASPPLite, metaclass=_RASPPMeta):
    pass


class RASPPLite3d(_RASPPLite, metaclass=_RASPPMeta):
    pass
