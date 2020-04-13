#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Tuple, Union

import torch.nn as nn
import torch.nn.functional as F

from combustion.util import double, single, triple


# type hints
Kernel2D = Union[Tuple[int, int], int]
Kernel3D = Union[Tuple[int, int, int], int]
Pad2D = Union[Tuple[int, int], int]
Pad3D = Union[Tuple[int, int, int], int]
Head = Union[bool, nn.Module]


class SpatialMeta(type):
    def __new__(cls, name, bases, dct):
        x = super().__new__(cls, name, bases, dct)
        if "3d" in name:
            x._dim = 3
            x._tuple = staticmethod(triple)
            x._xconv = F.conv_transpose3d
            x._conv = F.conv3d
            x._maxpool = F.max_pool3d
            x._maxunpool = F.max_unpool3d
            x._avgpool = F.avg_pool3d
            x._adapt_maxpool = F.adaptive_max_pool3d
            x._adapt_avgpool = F.adaptive_avg_pool3d
            x._lp_pool = None
            x._dropout = F.dropout
        elif "2d" in name:
            x._dim = 2
            x._tuple = staticmethod(double)
            x._xconv = F.conv_transpose2d
            x._conv = F.conv2d
            x._maxpool = F.max_pool2d
            x._maxunpool = F.max_unpool2d
            x._avgpool = F.avg_pool2d
            x._adapt_maxpool = F.adaptive_max_pool2d
            x._adapt_avgpool = F.adaptive_avg_pool2d
            x._lp_pool = F.lp_pool2d
            x._dropout = F.dropout2d
        elif "1d" in name:
            x._dim = 1
            x._tuple = staticmethod(single)
            x._xconv = F.conv_transpose1d
            x._conv = F.conv1d
            x._maxpool = F.max_pool1d
            x._maxunpool = F.max_unpool1d
            x._avgpool = F.avg_pool1d
            x._adapt_maxpool = F.adaptive_max_pool1d
            x._adapt_avgpool = F.adaptive_avg_pool1d
            x._lp_pool = F.lp_pool1d
            x._dropout = F.dropout3d
        else:
            raise RuntimeError(f"Metaclass: error processing name {cls.__name__}")
        x._transposed = "Transpose" in name
        x._act = nn.ReLU
        return x
