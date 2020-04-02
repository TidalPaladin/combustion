#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Provides command line flags to customize the training pipleine"""

import argparse
import os
import sys
import time
from argparse import ArgumentParser

from .validators import NumericValidator


def add_vision_args(parser: ArgumentParser) -> ArgumentParser:
    group = parser.add_argument_group("vision")

    group.add_argument("--power_two_crop", default=False, action="store_true", help="crop inputs to a power of two")
    group.add_argument("--invert", default=False, action="store_true", help="invert input images")
    group.add_argument("--hflip", default=False, action="store_true", help="randomly flip inputs horizontally")
    group.add_argument("--vflip", default=False, action="store_true", help="randomly flip inputs vertically")
    group.add_argument(
        "--rotate",
        type=int,
        low=0,
        high=180,
        inclusive=(True, True),
        default=0.0,
        metavar="DEG",
        action=NumericValidator,
        help="randomly rotate images up to +-DEG degrees",
    )
    group.add_argument(
        "--brightness",
        type=float,
        low=0.0,
        high=1.0,
        inclusive=(True, True),
        default=0.0,
        metavar="VAL",
        action=NumericValidator,
        help="randomly perturb brightness by +-VAL",
    )

    return parser
