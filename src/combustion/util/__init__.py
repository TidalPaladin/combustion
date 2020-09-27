#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .decorators import input, output
from .plot import alpha_blend, apply_colormap
from .util import Dim, double, ntuple, one_diff_tuple, replace_tuple, single, triple
from .validation import (
    check_dimension,
    check_dimension_match,
    check_is_array,
    check_is_tensor,
    check_names,
    check_names_match,
    check_ndim_match,
    check_shape,
    check_shapes_match,
)


__all__ = [
    "alpha_blend",
    "apply_colormap",
    "check_is_tensor",
    "check_names",
    "check_names_match",
    "check_ndim_match",
    "check_shape",
    "check_shapes_match",
    "check_dimension",
    "check_dimension_match",
    "check_is_array",
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
