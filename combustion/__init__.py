#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging

from .version import __version__


logger = logging.getLogger(name="combustion")


__all__ = ["logger", "__version__"]
