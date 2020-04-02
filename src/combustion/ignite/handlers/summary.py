#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import os
from typing import Any, Callable, Optional, Tuple, Union

from ignite.engine import Engine, Events
from torch import Tensor
from torch.nn import Module

try:
    from pytorch_model_summary import summary
except ImportError:
    raise ImportError(("Module requires pytorch_model_summary, install it" " with: pip install pytorch-model-summary"))


PrintFunc = Optional[Callable[[str], None]]
TransformFunc = Optional[Callable[[Any], Union[Tensor, Tuple[Tensor, ...]]]]


class SummaryWriter:
    r"""Prints and optionally saves a Keras style model summary

    Args:
        model (nn.Module): Model to summarize
        transform (transform, optional): Function to transform a batch to a model input
        filepath (str, optional): Filepath to save summary
        log_fn (callable, optional): Function to log the summary
        print_fn (callable, optional): Function to print the summary
        overwrite (bool, optional): If true, overwrite an existing conflicting file

    Keyword Args:
        Forwarded to summary call
    """

    def __init__(self, model, transform=None, filepath=None, log_fn=None, print_fn=None, overwrite=False, **kwargs):
        # type: (Module, TransformFunc, str, PrintFunc, PrintFunc, Optional[bool], ...)
        self.model = model
        self.transform = transform
        self.filepath = filepath
        self.log_fn = log_fn
        self.print_fn = print_fn
        self.overwrite = overwrite
        self.kwargs = kwargs

    def __repr__(self):
        # type: () -> str
        name = self.__class__.__name__
        model_name = self.model.__class__.__name__
        s = f"{name}({model_name}"
        if self.filepath:
            s += f", filepath={self.filepath}"
        if self.log_fn:
            s += f", log_fn={self.log_fn}"
        if self.print_fn:
            s += f", print_fn={self.print_fn}"
        return s + ")"

    def __call__(self, engine):
        # type: Engine
        input = engine.state.batch
        if self.transform is not None:
            input = self.transform(input)

        input.to(next(self.model.parameters()).device)
        model_summary = summary(self.model, input, **self.kwargs)

        if self.log_fn:
            self.log_fn("Model summary:\n%s", model_summary)
        if self.print_fn:
            self.print_fn(model_summary)
        if self.filepath:
            if not self.overwrite and os.path.isfile(self.filepath):
                raise FileExistsError(f"file {self.filepath} exists and overwrite=False")
            with open(self.filepath, "w") as f:
                for line in model_summary:
                    f.write(line)
        logging.debug("Handled model summary writing")

    @classmethod
    def from_args(cls, args, model, transform=None, filename="model_summary.txt", show_hierarchical=True):
        summary_path = os.path.join(args.result_path, filename)
        if not os.path.isdir(args.result_path):
            os.makedirs(args.result_path)

        return cls(
            model,
            transform=transform,
            filepath=summary_path,
            log_fn=logging.info,
            overwrite=args.overwrite,
            show_hierarchical=show_hierarchical,
        )

    def attach(self, engine):
        # type: (Engine)
        event = Events.ITERATION_COMPLETED(once=1)
        engine.add_event_handler(event, self)
