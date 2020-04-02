#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
import random
from argparse import Namespace
from typing import List, Optional, Tuple, Union

import torch
import torchvision.transforms.functional as xform
from PIL.Image import Image

from ..util.pytorch import input


@input("frames", name=("D", "C", "H", "W"), optional=True, drop_names=True)
@input("labels", name=("D", "C", "H", "W"), optional=True, drop_names=True)
def preprocess(
    *, frames: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None, args: Namespace, seed: int
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    if frames is None and labels is None:
        raise ValueError("frames and labels cannot both be None")
    if frames is not None and labels is not None and frames.shape[1:-2] != labels.shape[1:-2]:
        raise ValueError("frames.shape must match labels.shape: %s vs %s" % (frames.shape, labels.shape))

    result = []
    if frames is not None:
        frames = torch.stack([preprocess_frame(f, args, seed, False) for f in frames], 0)
        result.append(frames)
    if labels is not None:
        labels = torch.stack([preprocess_frame(f, args, seed, True) for f in labels], 0)
        result.append(labels)
    return tuple(result) if len(result) > 1 else result[0]


@input("frame", name=("C", "H", "W"), drop_names=True)
def preprocess_frame(frame: torch.Tensor, args: Namespace, seed: int, label: bool,) -> torch.Tensor:
    """preprocess_frame
    Runs the preprocessing pipeline based on supplied argparse args

    :param frame: The input frame to preprocess
    :type frame: torch.Tensor
    :param args: The argparse namespace with controlling flags
    :type args: Namespace
    :param seed: Seed value for deterministic preprocessing
    :type seed: int
    :param label: Boolean indicating if a segmentation label is being preprocessed
    :type label: bool

    :rtype: torch.Tensor The preprocessed frame
    """
    r = random.Random()
    r.seed(seed)

    # set PIL image mode based on input data type
    if label:
        frame.mul_(255)
        mode = "L"  # 8 bit pixel
    elif frame.dtype == torch.int16:
        mode = "I;16"  # 16 bit unsigned int
    else:
        mode = "L"  # 8 bit pixel

    _ = xform.to_pil_image(frame, mode=mode)

    # Crop/rotate applied to image and label
    if args.power_two_crop:
        _ = power_of_two_crop(_)
    if args.hflip and r.random() > 0.5:
        _ = xform.hflip(_)
    if args.vflip and r.random() > 0.5:
        _ = xform.vflip(_)
    if args.rotate != 0:
        low, high = -1 * args.rotate, args.rotate
        val = r.randrange(low, high)
        _ = xform.rotate(_, val)

    # early return before input only perturbations
    if label:
        _ = xform.to_tensor(_)
        result = torch.zeros_like(_)
        result[_ > 0] = 1
        return result.float()

    # random brightness perturbations
    if args.brightness:
        low = int(100 * (1.0 - args.brightness))
        high = int(100 * (1.0 + args.brightness))
        brightness = r.randrange(low, high) / 100.0
        _ = xform.adjust_brightness(_, brightness)

    # convert back to tensor for tensor ops
    _ = xform.to_tensor(_).float()
    if args.normalize:
        reduce_dims = (1, 2)
        mean = _.mean(dim=reduce_dims)
        std = torch.max(_.std(dim=reduce_dims), torch.Tensor([1e-6]))
        xform.normalize(_, mean, std, inplace=True)

    # random inversion
    if args.invert and r.random() > 0.5:
        _.mul_(-1)

    return _


def power_of_two_crop(frame: Union[Image, torch.Tensor]) -> Image:
    """power_of_two_crop
    Crops a frame to a power of two length along both the row
    and column dimensions. Cropping is performed such that the
    input image is preserved as much as possible.

    :param frame: the frame to be cropped. dimensions should be of the form CxHxW
    :type frame: torch.Tensor
    :rtype: torch.Tensor
    """
    if isinstance(frame, Image):
        height, width = frame.size
    else:
        frame = torch.as_tensor(frame)
        if not _is_2d_image(frame):
            raise ValueError("frame should be 2d image")
        height, width = frame.shape[-1], frame.shape[-2]
        frame = xform.to_pil_image(frame.float())

    crop_h = 2 ** math.floor(math.log(height, 2))
    crop_w = 2 ** math.floor(math.log(width, 2))
    return xform.center_crop(frame, (crop_h, crop_w))


@input("label", name=("N", "C", "H", "W"), drop_names=True)
def get_class_weights(label: torch.Tensor) -> torch.Tensor:
    """get_class_weights
    Generates a map of weights for segmentation labels. Weights
    are generated such that hot pixels have unit weight, and 
    dead pixel weights are scaled up or down according to the 
    ratio of hot to dead pixels.

    :param label:
    :type label: torch.Tensor
    :rtype: torch.Tensor

    :returns: Map of weights of same shape as `label`

    :Example:
        [[1, 0], [0, 0]] -> [[1, 1/3], [1/3, 1/3]]
    """
    return torch.stack([_get_class_weights(x) for x in label], 0)


def _get_class_weights(label):
    # pixel counts
    with torch.no_grad():
        total_pixel_count = label.numel()
        hot_pixel_count = label.sum().float()
        empty_pixel_count = total_pixel_count - hot_pixel_count

        # scale empty pixel weights down based on hot/empty ratio
        weight_empty = hot_pixel_count / empty_pixel_count
        weights = label.clone().detach().float()
        weights[~label.bool()] = weight_empty
        return weights


def _is_2d_image(frame):
    rank = len(frame.shape)
    if rank == 2:
        return True
    elif rank == 3 and frame.shape[0] <= 3:
        return True
    return False
