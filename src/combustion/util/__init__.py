#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .decorators import input, output
from .util import Dim, double, ntuple, one_diff_tuple, replace_tuple, single, triple
from .validation import (
    check_is_tensor,
    check_names,
    check_names_match,
    check_ndim_match,
    check_shape,
    check_shapes_match,
)


__all__ = [
    "check_is_tensor",
    "check_names",
    "check_names_match",
    "check_ndim_match",
    "check_shape",
    "check_shapes_match",
    "input",
    "output",
    "Dim",
    "one_diff_tuple",
    "ntuple",
    "single",
    "double",
    "triple",
    "replace_tuple",
]
