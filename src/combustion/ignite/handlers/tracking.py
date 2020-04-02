#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import os
import warnings
from argparse import Namespace
from typing import Any, Callable, Optional, Tuple

import matplotlib.pyplot as plt
from ignite.engine import Engine, Events
from torch import Tensor


class Tracker:
    def __call__(self, engine, metrics):
        if not hasattr(engine.state, "tracked"):
            setattr(engine.state, "tracked", {})

        x = (engine.state.iteration, engine.state.epoch)
        for name in metrics:
            if name not in engine.state.tracked:
                engine.state.tracked.setdefault(name, [])
            y = engine.state.metrics[name]
            engine.state.tracked[name].append((*x, y))

    def attach(self, engine, event, metrics, *args, **kwargs):
        engine.add_event_handler(event, self, metrics)
