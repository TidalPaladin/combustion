#!/usr/bin/env python
# -*- coding: utf-8 -*-

from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class OCR(nn.Module):
    r"""Implements Object Contextual Representations (OCR) as described in
    `Object-Contextual Representations for Semantic Segmentation`_. In problem domains where
    the label of a pixel is the category of object to which that pixel belongs, OCR refines
    pixel level predictions be exploiting the representation of the corresponding object class.

    .. image:: ./ocr.png
        :width: 800px
        :align: center
        :height: 250px
        :alt: OCR Pipeline.

    Note that the soft object regions are learned under the supervision of ground truth labels.

    Args:
        in_channels (int):
            The number of input channels :math:`C_i`. Both the pixel representations and soft
            object regions are expected to have :math:`C_i` channels.

        key_channels (int):
            Number of channels in the attention mechanism.

        out_channels (int):
            Number of channels in the produced augmented representations, :math:`C_o`.

        downsample (int):
            An integer factor by which to downsample the pixel representations to conserve
            memory. Downsampling is performed via :class:`torch.nn.MaxPool2d` and the output
            will be upsampled to it's original shape via a bilinear interpolation.

        activation (:class:`torch.nn.Module`):
            Activation function to use for each convolution.

        align_corners (bool):
            See :func:`torch.nn.functional.interpolate`. Only has an effect when ``downsample > 1``.

    Shapes:
        * ``pixels`` - :math:`(N, C_i, H, W)`
        * ``regions`` - Same as ``pixels``
        * Output - :math:`(N, C_o, H, W)`

    .. _Object-Contextual Representations for Semantic Segmentation:
        https://arxiv.org/abs/1909.11065
    """

    def __init__(
        self,
        in_channels: int,
        key_channels: int,
        out_channels: int,
        downsample: int = 1,
        activation: nn.Module = nn.ReLU(),
        align_corners: bool = True,
    ):
        super().__init__()
        # validation
        downsample = abs(int(downsample))
        if downsample < 1:
            raise ValueError(f"Expected downsample >= 1 but found {downsample}")
        self.align_corners = align_corners

        # downsampling via max pool to save memory
        if downsample > 1:
            self.downsample = nn.MaxPool2d(downsample)
        else:
            self.downsample = nn.Identity()

        # query produced from pixel inputs
        self.pixel_conv = nn.Sequential(
            nn.Conv2d(in_channels, key_channels, 1, bias=False),
            nn.BatchNorm2d(key_channels),
            deepcopy(activation),
            nn.Conv2d(key_channels, key_channels, 1, bias=False),
            nn.BatchNorm2d(key_channels),
            deepcopy(activation),
        )

        # key produced from soft object regions
        self.object_key_conv = nn.Sequential(
            nn.Conv2d(in_channels, key_channels, 1, bias=False),
            nn.BatchNorm2d(key_channels),
            deepcopy(activation),
            nn.Conv2d(key_channels, key_channels, 1, bias=False),
            nn.BatchNorm2d(key_channels),
            deepcopy(activation),
        )

        # value produced from soft object regions
        self.object_value_conv = nn.Sequential(
            nn.Conv2d(in_channels, key_channels, 1, bias=False),
            nn.BatchNorm2d(key_channels),
            deepcopy(activation),
        )

        # conv for context tensor produced from attention
        self.context_conv = nn.Sequential(
            nn.Conv2d(key_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            deepcopy(activation),
        )

        # final conv that produces augmented pixel representations
        # input is a concat of pixel representations and object contextual representations
        self.augmentation_conv = nn.Sequential(
            nn.Conv2d(2 * in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            deepcopy(activation),
        )

    def forward(self, pixels: Tensor, regions: Tensor) -> Tensor:
        batch_size, num_classes, height, width = pixels.shape
        pixels = self.downsample(pixels)

        # get query tensor, permuted to shape (batch_size, -1, key_channels)
        query = self.pixel_conv(pixels)
        key_channels = query.shape[1]
        query = query.view(batch_size, key_channels, -1).permute(0, 2, 1)

        # get key tensor of shape (batch_size, key_channels, -1)
        key = self.object_key_conv(regions).view(batch_size, key_channels, -1)

        # get value tensor of shape (batch_size, -1, key_channels)
        value = self.object_value_conv(regions).view(batch_size, key_channels, -1)
        value = value.permute(0, 2, 1)

        # (N, S, K) X (N, K, S) => (N, S, S)
        attention = torch.bmm(query, key)
        attention = F.softmax(attention, dim=-1)

        # (N, S, S) X (N, S, K) => (N, S, K)
        context = torch.bmm(attention, value)

        # restore original shape (S => H, W) to produce (N, K, H, W)
        context = context.permute(0, 2, 1).view(batch_size, key_channels, pixels.shape[2], pixels.shape[3])
        context = self.context_conv(context)

        # undo max pooling if needed
        if context.shape[2:] != torch.Size([height, width]):
            context = F.interpolate(context, size=(height, width), mode="bilinear", align_corners=self.align_corners)

        # concat pixel representation with object contextual representation,
        # pass through final conv to produced augmented pixel representation
        _ = torch.cat([pixels, context], dim=1)
        output = self.augmentation_conv(_)
        return output

    @staticmethod
    def create_region_target(categorical_target: Tensor, num_classes: int) -> Tensor:
        r"""Creates a soft object region target given a categorical segmentation target.
        The result will be a tensor of shape :math:`(N, C, *)` where :math:`C_i` is
        the one hot region target for class :math:`i`.

        Args:

            categorical_target (:class:`Tensor`):
                Categorical segmentation target to convert into a one-hot binary soft object region target

            num_classes (int):
                Number of classes in the segmentation problem, :math:`C`

        Shapes:
            * ``categorical_target`` - :math:`(N, *)`
            * Output - :math:`(N, C, *)` where :math:`C` is ``num_classes``
        """
        categorical_target.ndim
        one_hot = F.one_hot(categorical_target, num_classes).unsqueeze_(1).transpose(1, -1).squeeze_(-1).contiguous()
        return one_hot
