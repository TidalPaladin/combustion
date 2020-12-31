#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch

from combustion.vision import mask_to_polygon


def test_mask_to_polygon():
    x = torch.tensor(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 2, 2],
            [0, 0, 0, 0, 2, 2],
        ]
    )
    x = x.view(1, 1, *x.shape)
    num_classes = 3
    result = mask_to_polygon(x, num_classes)
    assert isinstance(result, tuple)
    assert isinstance(result[0], tuple)
    assert isinstance(result[0][0], torch.Tensor)
