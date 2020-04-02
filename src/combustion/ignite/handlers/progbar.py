#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import os
from argparse import Namespace
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Union

import ignite.handlers as handlers
import torch
from ignite.contrib.handlers.tqdm_logger import ProgressBar as _ProgressBar
from ignite.contrib.metrics import GpuInfo
from ignite.engine import Engine, Events
from torch import Tensor
from torch.nn import Module


class ProgressBar(_ProgressBar):
    def __init__(self, *args, **kwargs):
        super(ProgressBar, self).__init__(*args, **kwargs)
        self.gpu_info = None
        self.metrics = None

    @classmethod
    def from_args(cls, args, metrics=[]):
        if not args.progbar:
            return None

        if args.progbar_format is None:
            progbar = cls()
        else:
            progbar = cls(bar_format=args.progbar_format)

        if args.gpuinfo:
            progbar.gpu_info = GpuInfo()
            metrics.append(args.gpu_format)
        if args.lr_decay is not None:
            metrics.append("lr")

        progbar.metrics = metrics
        return progbar

    def attach(self, engine):
        # type: (Engine)
        if self.gpu_info is not None:
            self.gpu_info.attach(engine, name="gpu")
        super(ProgressBar, self).attach(engine, self.metrics)
