#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
from typing import Optional

import torch
import torch.nn as nn
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import Engine, Events
from torch.utils.data import DataLoader


class ValidationLogger:
    r"""Logs validation results to standard output and a ProgressBar if desired

    .. note::

        Arg `fmt` can formatted according to any attribute of `engine.state` 
        or `engine.state.metrics`.

    Args:
        fmt (str): Format for the logged message
        progbar (ProgressBar): A progress bar to log the message to
    """

    def __init__(self, fmt, progbar=None):
        # type: (str, Optional[ProgressBar])
        self.progbar = progbar
        self.fmt = fmt

    def __call__(self, engine):
        # type: (Engine)
        fmt_keys = engine.state
        fmt_keys = vars(engine.state).copy()
        fmt_keys.update(engine.state.metrics)
        msg = self.fmt.format(**fmt_keys)

        logging.info(msg)
        if self.progbar is not None:
            self.progbar.log_message(msg)

    def attach(self, engine):
        # type: (Engine)
        r"""
        Attaches the logger task to a validation engine as an `Events.EPOCH_COMPLETED` event.

        Args:
            engine (Engine): The validation engine to attach to
        """
        event = Events.EPOCH_COMPLETED
        engine.add_event_handler(event, self)


class Validate:
    r"""Validation handler that deals with validation epoch counting nicely.

    .. note::

        The validation engine run will be wrapped in `torch.no_grad()`

    Args:
        engine (Engine): Validation engine
        data (DataLoader): Loader for validation dataset
        model (nn.Module, optional): If given, `model.eval()` will be automatically called
    """

    def __init__(self, engine, data, model=None):
        # type: (Engine, DataLoader, Optional[nn.Module])
        engine.add_event_handler(Events.EPOCH_COMPLETED, lambda x: x.terminate())
        self.engine = engine
        self.data = data
        self.model = model

    def __call__(self, engine):
        # type: (Engine)
        if self.model is not None:
            self.model.eval()
        with torch.no_grad():
            self.engine.run(self.data, max_epochs=engine.state.max_epochs)

    def attach(self, engine):
        # type: (Engine)
        r"""
        Attaches the validation engine as an `Events.EPOCH_COMPLETED` event.

        Args:
            engine (Engine): The training engine to attach to
        """
        event = Events.EPOCH_COMPLETED
        engine.add_event_handler(event, self)
