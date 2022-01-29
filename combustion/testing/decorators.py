#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest

from .utils import cuda_available


cuda_or_skip = pytest.mark.skipif(not cuda_available(), reason="cuda not available")
cuda_or_skip.__doc__ = r"""
    Run test only if :func:`torch.cuda.is_available()` is true.
"""
