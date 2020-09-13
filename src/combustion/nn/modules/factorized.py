#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module, init
from torch.nn.parameter import Parameter

from combustion.util import double, one_diff_tuple, single, triple

from .util import SpatialMeta


__all__ = [
    "Conv3d",
    "ConvTranspose3d",
    "Conv2d",
    "ConvTranspose2d",
    "Conv1d",
    "ConvTranspose1d",
]


class FactorizedMeta(SpatialMeta):
    def __new__(cls, name, bases, dct):
        name = "Factorized" + name
        x = super().__new__(cls, name, bases, dct)
        if "3d" in name:
            x._xconv = staticmethod(conv_transpose3d)
            x._conv = staticmethod(conv3d)
        elif "2d" in name:
            x._xconv = staticmethod(conv_transpose2d)
            x._conv = staticmethod(conv2d)
        elif "1d" in name:
            x._xconv = staticmethod(conv_transpose1d)
            x._conv = staticmethod(conv1d)
        else:
            raise RuntimeError(f"Metaclass: error processing name {cls.__name__}")
        return x


class _ConvNd(Module):

    __constants__ = [
        "stride",
        "padding",
        "dilation",
        "groups",
        "bias",
        "padding_mode",
        "output_padding",
        "in_channels",
        "out_channels",
        "kernel_size",
    ]

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        output_padding,
        groups,
        bias,
        padding_mode,
    ):
        super(_ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = self._tuple(kernel_size)
        self.stride = self._tuple(stride)
        self.padding = self._tuple(padding)
        self.dilation = self._tuple(dilation)
        self.output_padding = self._tuple(output_padding)
        self.groups = groups
        self.padding_mode = padding_mode

        _pw_kernel = (1,) * self._dim

        self.pointwise = Parameter(torch.Tensor(out_channels, in_channels // groups, *_pw_kernel))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.spatial = []
        for x in range(self._dim):
            kernel = one_diff_tuple(self._dim, 1, self.kernel_size[x], x)
            weight = Parameter(torch.Tensor(out_channels, 1, *kernel))
            setattr(self, f"spatial_{x}", weight)
            self.spatial.append(getattr(self, f"spatial_{x}"))

        self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.named_parameters(prefix="", recurse=True):
            if name != "bias":
                init.kaiming_uniform_(param, a=math.sqrt(5))
                if self.bias is not None and name == "pointwise":
                    fan_in, _ = init._calculate_fan_in_and_fan_out(param)
                    bound = 1 / math.sqrt(fan_in)
                    init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        s = "{in_channels}, {out_channels}, kernel_size={kernel_size}" ", stride={stride}"
        if self.padding != (0,) * len(self.padding):
            s += ", padding={padding}"
        if self.dilation != (1,) * len(self.dilation):
            s += ", dilation={dilation}"
        if self.output_padding != (0,) * len(self.output_padding):
            s += ", output_padding={output_padding}"
        if self.groups != 1:
            s += ", groups={groups}"
        if self.bias is None:
            s += ", bias=False"
        if self.padding_mode != "zeros":
            s += ", padding_mode={padding_mode}"
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super(_ConvNd, self).__setstate__(state)
        if not hasattr(self, "padding_mode"):
            self.padding_mode = "zeros"

    def forward(self, input, output_size=None):
        # type: (Tensor, Optional[List[int]]) -> Tensor
        if self._transposed and self.padding_mode != "zeros":
            raise ValueError("Only `zeros` padding mode is supported for {}".format(self.__class__.__name__))

        if output_size is not None:
            output_padding = self._output_padding(input, output_size, self.stride, self.padding, self.kernel_size)
            return self._xconv(
                input,
                self.pointwise,
                self.spatial,
                self.bias,
                self.stride,
                self.padding,
                output_padding,
                self.groups,
                self.dilation,
            )

        if self.padding_mode == "circular":
            expanded_padding = tuple(
                reversed([((self.padding[i] + 1) // 2, self.padding[i] // 2) for i in range(self._dim + 1)])
            )
            input = F.pad(input, expanded_padding, mode="circular")
            padding = 0
        else:
            padding = self.padding

        if self._transposed:
            return self._xconv(
                input,
                self.pointwise,
                self.spatial,
                None,
                self.stride,
                padding,
                self.output_padding,
                self.groups,
                self.dilation,
            )
        else:
            return self._conv(
                input,
                self.pointwise,
                self.spatial,
                None,
                self.stride,
                padding,
                self.dilation,
                self.groups,
            )


def conv1d(
    input,
    pointwise,
    spatial,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
):
    if not isinstance(spatial, Tensor):
        spatial = spatial[0]
    stride = single(stride)
    padding = single(padding)
    dilation = single(dilation)
    _ = F.conv1d(input, pointwise, bias, 1, 0, 1, groups)
    _ = F.conv1d(_, spatial, None, stride, padding, dilation, _.shape[1])
    return _


def conv2d(
    input,
    pointwise,
    spatial,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
):
    stride = double(stride)
    padding = double(padding)
    dilation = double(dilation)
    _ = F.conv2d(input, pointwise, bias, 1, 0, 1, groups)
    for i, weight in enumerate(spatial):
        stri = one_diff_tuple(2, 1, stride[i], i)
        pad = one_diff_tuple(2, 0, padding[i], i)
        dil = one_diff_tuple(2, 1, dilation[i], i)
        _ = F.conv2d(_, weight, None, stri, pad, dil, _.shape[1])
    return _


def conv3d(
    input,
    pointwise,
    spatial,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
):
    stride = triple(stride)
    padding = triple(padding)
    dilation = triple(dilation)
    _ = F.conv3d(input, pointwise, bias, 1, 0, 1, groups)
    for i, weight in enumerate(spatial):
        stri = one_diff_tuple(3, 1, stride[i], i)
        pad = one_diff_tuple(3, 0, padding[i], i)
        dil = one_diff_tuple(3, 1, dilation[i], i)
        _ = F.conv3d(_, weight, None, stri, pad, dil, _.shape[1])
    return _


def conv_transpose1d(
    input,
    pointwise,
    spatial,
    bias=None,
    stride=1,
    padding=0,
    output_padding=0,
    groups=1,
    dilation=1,
):
    if not isinstance(spatial, Tensor):
        spatial = spatial[0]
    stride = single(stride)
    padding = single(padding)
    output_padding = single(output_padding)
    dilation = single(dilation)
    _ = F.conv1d(input, pointwise, bias, 1, 0, 1, groups)
    _ = F.conv_transpose1d(_, spatial, None, stride, padding, output_padding, _.shape[1], dilation)
    return _


def conv_transpose2d(
    input,
    pointwise,
    spatial,
    bias=None,
    stride=1,
    padding=0,
    output_padding=0,
    groups=1,
    dilation=1,
):
    stride = double(stride)
    padding = double(padding)
    output_padding = double(output_padding)
    dilation = double(dilation)
    _ = F.conv2d(input, pointwise, bias, 1, 0, 1, groups)
    for i, weight in enumerate(spatial):
        stri = one_diff_tuple(2, 1, stride[i], i)
        pad = one_diff_tuple(2, 0, padding[i], i)
        out_pad = one_diff_tuple(2, 0, output_padding[i], i)
        dil = one_diff_tuple(2, 1, dilation[i], i)
        _ = F.conv_transpose2d(_, weight, None, stri, pad, out_pad, _.shape[1], dil)
    return _


def conv_transpose3d(
    input,
    pointwise,
    spatial,
    bias=None,
    stride=1,
    padding=0,
    output_padding=0,
    groups=1,
    dilation=1,
):
    stride = triple(stride)
    padding = triple(padding)
    output_padding = triple(output_padding)
    dilation = triple(dilation)
    _ = F.conv3d(input, pointwise, bias, 1, 0, 1, groups)
    for i, weight in enumerate(spatial):
        stri = one_diff_tuple(3, 1, stride[i], i)
        pad = one_diff_tuple(3, 0, padding[i], i)
        out_pad = one_diff_tuple(3, 0, output_padding[i], i)
        dil = one_diff_tuple(3, 1, dilation[i], i)
        _ = F.conv_transpose3d(_, weight, None, stri, pad, out_pad, _.shape[1], dil)
    return _


class _ConvTransposeMixin(object):
    def _output_padding(self, input, output_size, stride, padding, kernel_size):
        # type: (Tensor, Optional[List[int]], List[int], List[int], List[int]) -> List[int]
        if output_size is None:
            ret = single(self.output_padding)  # converting to list if was not already
        else:
            k = input.dim() - 2
            if len(output_size) == k + 2:
                output_size = output_size[2:]
            if len(output_size) != k:
                raise ValueError("output_size must have {} or {} elements (got {})".format(k, k + 2, len(output_size)))

            min_sizes = torch.jit.annotate(List[int], [])
            max_sizes = torch.jit.annotate(List[int], [])
            for d in range(k):
                dim_size = (input.size(d + 2) - 1) * stride[d] - 2 * padding[d] + kernel_size[d]
                min_sizes.append(dim_size)
                max_sizes.append(min_sizes[d] + stride[d] - 1)

            for i in range(len(output_size)):
                size = output_size[i]
                min_size = min_sizes[i]
                max_size = max_sizes[i]
                if size < min_size or size > max_size:
                    raise ValueError(
                        (
                            "requested an output size of {}, but valid sizes range "
                            "from {} to {} (for an input of {})"
                        ).format(output_size, min_sizes, max_sizes, input.size()[2:])
                    )

            res = torch.jit.annotate(List[int], [])
            for d in range(k):
                res.append(output_size[d] - min_sizes[d])

            ret = res
        return ret


class Conv3d(_ConvNd, metaclass=FactorizedMeta):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
    ):
        super(Conv3d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            0,
            groups,
            bias,
            padding_mode,
        )

    def forward(self, input):
        return super(Conv3d, self).forward(input)


class ConvTranspose3d(_ConvTransposeMixin, _ConvNd, metaclass=FactorizedMeta):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        groups=1,
        bias=True,
        dilation=1,
        padding_mode="zeros",
    ):
        super(ConvTranspose3d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            output_padding,
            groups,
            bias,
            padding_mode,
        )

    def forward(self, input):
        return super(ConvTranspose3d, self).forward(input)


class Conv2d(_ConvNd, metaclass=FactorizedMeta):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
    ):
        super(Conv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            0,
            groups,
            bias,
            padding_mode,
        )

    def forward(self, input):
        return super(Conv2d, self).forward(input)


class ConvTranspose2d(_ConvTransposeMixin, _ConvNd, metaclass=FactorizedMeta):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        groups=1,
        bias=True,
        dilation=1,
        padding_mode="zeros",
    ):
        super(ConvTranspose2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            output_padding,
            groups,
            bias,
            padding_mode,
        )

    def forward(self, input):
        return super(ConvTranspose2d, self).forward(input)


class Conv1d(_ConvNd, metaclass=FactorizedMeta):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
    ):
        super(Conv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            0,
            groups,
            bias,
            padding_mode,
        )

    def forward(self, input):
        return super(Conv1d, self).forward(input)


class ConvTranspose1d(_ConvTransposeMixin, _ConvNd, metaclass=FactorizedMeta):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        groups=1,
        bias=True,
        dilation=1,
        padding_mode="zeros",
    ):
        super(ConvTranspose1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            output_padding,
            groups,
            bias,
            padding_mode,
        )

    def forward(self, input):
        return super(ConvTranspose1d, self).forward(input)
