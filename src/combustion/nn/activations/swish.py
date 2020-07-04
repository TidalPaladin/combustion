#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# implementation inspired by
# https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/utils.py


class _SwishFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        for i in ctx.saved_tensors:
            sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


def swish(inputs: Tensor, memory_efficient: bool = True) -> Tensor:
    r"""The swish activation function, defined as

    .. math::
        f(x) = x \cdot \text{sigmoid}(x)

    Args:

        inputs (Tensor):
            The input tensor

        memory_efficient (bool, optional):
            Whether or not to use an implementation that is more memory efficient at training
            time. When ``memory_efficient=True``, this method is incompatible with TorchScript.

    .. warning::
        This method is traceable with TorchScript when ``memory_efficient=False``, but is
        un-scriptable due to the use of :class:`torch.autograd.Function` for a
        memory-efficient backward pass. Please export using :func:`torch.jit.trace` with
        ``memory_efficient=False``
    """
    if memory_efficient:
        return _SwishFunction.apply(inputs)
    else:
        return inputs * torch.sigmoid(inputs)


class Swish(nn.Module):
    r"""The swish activation function, defined as

    .. math::
        f(x) = x \cdot \text{sigmoid}(x)

    .. warning::
        This method is traceable with TorchScript, but is un-scriptable due to the
        use of :class:`torch.autograd.Function` for a memory-efficient backward pass.
        Please export using :func:`torch.jit.trace` after calling ``module.eval()``.
    """

    @torch.jit.ignore
    def _memory_efficient_forward(self, inputs: Tensor) -> Tensor:
        return swish(inputs)

    def forward(self, inputs: Tensor) -> Tensor:
        if not self.training:
            return self._memory_efficient_forward(inputs)
        else:
            return inputs * torch.sigmoid(inputs)


def hard_swish(inputs: Tensor, inplace: bool = False) -> Tensor:
    r"""The hard swish activation function proposed in
    `Searching For MobileNetV3`_, defined as

    .. math::
        f(x) = x \cdot \frac{\text{ReLU6}(x + 3)}{6}

    Hard swish approximates the swish activation, but computationally cheaper due to the
    removal of :math:`\text{sigmoid}(x)`.

    Args:

        inputs (Tensor):
            The input tensor

        inplace (bool, optional):
            Whether or not to perform the operation in place.

    .. _Searching for MobileNetV3:
        https://arxiv.org/abs/1905.02244
    """
    if inplace:
        return inputs.mul_(F.relu6(inputs + 3, inplace=True).div_(6))
    else:
        return F.relu6(inputs + 3).div(6).mul(inputs)


class HardSwish(nn.Module):
    r"""The hard swish activation function proposed in
    `Searching For MobileNetV3`_, defined as

    .. math::
        f(x) = x \cdot \frac{\text{ReLU6}(x + 3)}{6}

    Hard swish approximates the swish activation, but computationally cheaper due to the
    removal of :math:`\text{sigmoid}(x)`.


    .. image:: ./hswish.png
        :width: 600px
        :align: center
        :height: 300px
        :alt: Comparison of Hard Swish and Swish activations.

    Args:

        inplace (bool, optional):
            Whether or not to perform the operation in place.

    .. _Searching for MobileNetV3:
        https://arxiv.org/abs/1905.02244
    """

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def extra_repr(self):
        if self.inplace:
            return "inplace=True"
        else:
            return ""

    def forward(self, inputs: Tensor) -> Tensor:
        return hard_swish(inputs, self.inplace)
