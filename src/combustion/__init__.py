#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging

from .__main__ import MultiRunError, check_exceptions, clear_exceptions, initialize, main
from .version import __version__


logger = logging.getLogger(name="combustion")


__all__ = [
    "main",
    "MultiRunError",
    "check_exceptions",
    "clear_exceptions",
    "initialize",
    "logger",
    "__version__",
]
