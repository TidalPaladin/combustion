#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import os
from abc import ABC, abstractmethod
from argparse import Namespace
from inspect import isgenerator
from typing import Any, Callable, Optional, Tuple

import matplotlib.pyplot as plt
from ignite.engine import Engine, Events
from torch import Tensor


class Visualizer:
    def __init__(self, dir_format, file_format, process_fn, visualize_fn, title, overwrite, **kwargs):
        # type: (str, str, Callable[[Engine, Any], Tuple[Tensor, ...]], Callable[[Any, ...], None], Optional[str])
        self.dir_format = dir_format
        self.file_format = file_format
        self.title = title
        self.process_fn = process_fn if process_fn is not None else self.process
        self.visualize_fn = visualize_fn if visualize_fn is not None else self.visualize
        self.overwrite = overwrite
        self.kwargs = kwargs

    def __repr__(self):
        name = self.__class__.__name__
        path = os.path.join(self.dir_format, self.file_format)
        s = f"{name}({path}, title={self.title}"
        if self.overwrite:
            s += ", overwrite={self.overwrite}"
        s += ")"
        return s

    def visualize(self, *inputs, title):
        raise NotImplementedError("must implement visualize or pass visualize_fn in constructor")

    def process(self, engine):
        raise NotImplementedError("must implement visualize or pass process_fn in constructor")

    def _format(self, engine):
        fmt_keys = engine.state
        fmt_keys = vars(engine.state).copy()
        fmt_keys.update(engine.state.metrics)
        path = self.dir_format.format(**fmt_keys)
        filename = self.file_format.format(**fmt_keys)
        title = self.title.format(**fmt_keys)
        return path, filename, title

    def _get_filepath(self, path, filename):
        if not os.path.isdir(path):
            os.makedirs(path)
        filepath = os.path.join(path, filename)
        if not self.overwrite and os.path.isfile(filepath):
            raise FileExistsError(f"file {filepath} exists")
        return filepath

    def _savefig(self, filepath, fig, title):
        if isgenerator(fig):
            for i, target in enumerate(fig):
                name, ext = filepath.split(".")
                name += f"_sub_{i}.{ext}"
                target.savefig(name, **self.kwargs)
                plt.close(target)
        else:
            fig.savefig(filepath, **self.kwargs)
            plt.close(fig)

    def __call__(self, engine):
        # type: (Engine, Any) -> None
        inputs = self.process_fn(engine.state)
        inputs = (inputs,) if not isinstance(inputs, tuple) else inputs

        path, filename, title = self._format(engine)
        filepath = self._get_filepath(path, filename)
        inputs = tuple([x.clone().detach().cpu() if isinstance(x, Tensor) else x for x in inputs])
        fig = self.visualize_fn(*inputs, title)
        self._savefig(filepath, fig, title)


class OutputVisualizer(Visualizer):
    r"""Visualizes validation outputs.

    .. note::

        Args `dir_format`, `file_format`, and `title` can formatted according to any
        attribute of `engine.state` or `engine.state.metrics`

    Args:
        dir_format (str): Format for the directory in which images from an epoch will be placed
        file_format (str): Format for each image filename
        process_fn (callable): Function that maps `engine.state` to one or more Tensors 
        visualize_fn (callable): Function that accepts one or Tensors and an optional title. 
            Function should visualize the input tensors to the given file.
        title (str, optional): A formatted title to pass to `visualize_fn`

    Examples::

        >>> # Formats
        >>> dir_format = 'epoch_{epoch}'
        >>> file_format = 'output_{iteration}.png'
        >>> title_format = 'Validation output: epoch {epoch} iteration {iteration}'
        >>> vis = OutputVisualizer(dir_format, file_format, process_fn, visualize_fn, title_format)
        >>> engine.add_event_handler(Events.ITERATION_COMPLETED, val_engine)
    """

    def __init__(
        self, dir_format, file_format, process_fn=None, visualize_fn=None, title=None, overwrite=False, **kwargs
    ):
        # type: (str, str, Callable[[Engine, Any], Tuple[Tensor, ...]], Callable[[Any, ...], None], Optional[str])
        super().__init__(dir_format, file_format, process_fn, visualize_fn, title, overwrite, **kwargs)

    def attach(self, engine, event=Events.ITERATION_COMPLETED):
        engine.add_event_handler(event, self)

    @classmethod
    def from_args(cls, args, process_fn, visualize_fn, path=None, title=None, fmt=None, **kwargs):
        # type: (Any, Namespace, Callable[[Engine, Any], Tuple[Tensor, ...]], Callable[[Any, ...], None])
        r"""Constructs an OutputVisualizer based on an argparse Namespace.

        .. note::

            The `Namespace` should have attributes
            * :attr:`result_path` base path where  
            * :attr:`val_image_dir_fmt` passed to `dir_format`
            * :attr:`val_image_file_fmt` passed to `file_format`
            * :attr:`val_image_title` passed to `title`

        Args:
            args (Namespace): Argparse Namespace
            process_fn (callable): Function that maps `engine.state` to one or more Tensors 
            visualize_fn (callable): Function that accepts one or Tensors and an optional title. 
                Function should visualize the input tensors to the given file.
        """
        if path is None:
            path = os.path.join(args.result_path, args.val_image_dir_fmt)
        else:
            path = os.path.join(args.result_path, path)
        fmt = args.val_image_file_fmt if fmt is None else fmt
        title = args.val_image_title if title is None else title
        return cls(path, fmt, process_fn, visualize_fn, title, **kwargs)


class TrackedVisualizer(Visualizer):
    def __init__(self, dir_format, file_format, metrics, axis="step", title=None, **kwargs):
        self.metrics = metrics
        self.axis = axis
        super(TrackedVisualizer, self).__init__(dir_format, file_format, None, None, title, True, **kwargs)

    def __call__(self, engine, transform=None):
        # type: (Engine, Any) -> None
        path, filename, title = self._format(engine)
        filepath = self._get_filepath(path, filename)

        rows = min(len(self.metrics), 2)
        cols = len(self.metrics) // (rows + 1) + 1
        fig, axs = plt.subplots(rows, cols, figsize=(8, 6))
        fig.suptitle(title)
        fig.tight_layout()
        fig.subplots_adjust(left=0.1, bottom=0.1, top=0.9, hspace=0.3)

        for ax, metric in zip(axs.ravel(), self.metrics):
            values = engine.state.tracked[metric]
            iterations, epochs, y = list(zip(*values))
            if self.axis == "step":
                x = iterations
            else:
                x = epochs
            ax.set_xlabel(self.axis)
            ax.set_title(metric)
            ax.plot(x, y)
            ax.ticklabel_format(axis="both", style="sci", scilimits=(-2, 2))

        self._savefig(filepath, fig, title)

    def attach(self, engine, event=Events.EPOCH_COMPLETED):
        engine.add_event_handler(event, self)
