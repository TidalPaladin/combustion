#!/usr/bin/env python
# -*- coding: utf-8 -*-


from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Tuple, Union

import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

from .base import AttributeCallback, mkdir, resolve_dir


PyplotLogFunction = Callable[
    [str, plt.Figure, pl.Trainer, pl.LightningModule, int, Optional[int], AttributeCallback], None
]


class PyplotSave:
    r"""Log function for :class:`MatplotlibCallback` that saves figures PNG files.

    Args:
        path (:class:`Path`):
            Path where images will be saved. Defaults to ``trainer.default_root_dir``.

    Keyword Args:
        Forwarded to :func:`matplotlib.pyplot.Figure.savefig`
    """

    def __init__(self, path: Optional[Path] = None, **kwargs):
        self._path = Path(path) if path is not None else None
        self.path = None
        self.kwargs = kwargs

    @staticmethod
    def save_figure(path: Path, fig: plt.Figure, **kwargs) -> None:
        path = Path(path).with_suffix(".png")
        path.parent.mkdir(exist_ok=True, parents=True)
        fig.savefig(str(path), **kwargs)

    def __call__(
        self,
        name: str,
        fig: plt.Figure,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        step: int,
        batch_idx: int,
        caller: Callback,
    ) -> None:
        if self.path is None:
            self.path = Path(resolve_dir(trainer, self._path, "saved_figures"))

        root = Path(self.path)
        dest = Path(root, name, caller.read_step_as_str(pl_module, batch_idx)).with_suffix(".png")
        mkdir(dest.parent, trainer)
        self.save_figure(dest, fig)


def tensorboard_log_figure(
    name: str,
    fig: plt.Figure,
    trainer: pl.Trainer,
    pl_module: pl.LightningModule,
    step: int,
    batch_idx: Optional[int],
    caller: Callback,
) -> None:
    experiment = pl_module.logger.experiment
    experiment.add_figure(name, fig, step)


class MatplotlibCallback(AttributeCallback):
    r"""Callback for visualizing image tensors using a PyTorch Lightning logger.

    Example:
        >>> callback = VisualizeCallback("inputs")
        >>>
        >>> # LightningModule.training_step
        >>> def training_step(self, batch, batch_idx):
        >>>     image, target = batch
        >>>     ...
        >>>     # attribute will be logged to TensorBoardLogger under 'train/inputs'
        >>>     self.last_image = image

    Args:
        name (str):
            Figure name

        path (str):
            Directory path to save figure to

        triggers (str or list of str):
            Modes for which the callback should run. Must be one of
            ``"train"``, ``"val"``, ``"test"``.

        attr_name (str):
            Name of the attribute where the callback will search for the image to be logged.

        epoch_counter (bool):
            If ``True``, report the epoch for each callback invocation. By default, the
            global step is reported.

        max_calls (int, optional):
            If given, do not log more than ``max_calls`` batches per epoch.

        interval (int, optional):
            If given, only execute the callback every ``interval`` steps

        ignore_errors (bool):
            If ``True``, do not raise an exception if ``attr_name`` cannot be found.

        log_fn (:class:`PyplotLogFunction` or iterable of such functions):
            Callable(s) that logs/saves the plot.

        close (bool):
            If ``False``, do not close the figure after callback completes
    """

    def __init__(
        self,
        name: str,
        triggers: Union[str, Iterable[str]] = ("train", "val", "test"),
        hook: str = "step",
        attr_name: str = "last_image",
        epoch_counter: bool = False,
        max_calls: Optional[int] = None,
        interval: Optional[int] = None,
        ignore_errors: bool = False,
        log_fn: Union[PyplotLogFunction, Iterable[PyplotLogFunction]] = tensorboard_log_figure,
        close: bool = True,
    ):
        super().__init__(
            triggers,
            hook,
            attr_name,
            epoch_counter,
            max_calls,
            interval,
            ignore_errors,
        )
        self.name = str(name)
        self.close = bool(close)

        if isinstance(log_fn, Iterable):
            self.log_fn = log_fn
        elif isinstance(log_fn, Callable):
            self.log_fn = (log_fn,)
        elif log_fn is None:
            self.log_fn = (tensorboard_log_figure,)
        else:
            raise ValueError(f"Unknown `log_fn`: {log_fn}")

    def prepare_figure(
        self, hook: Tuple[str, str], attr: Any, trainer: pl.Trainer, pl_module: pl.LightningModule, step: int
    ) -> None:
        r"""Called before the callback executes. Does nothing by default, but can be overridden to prepare
        an arbitrary attribute into a loggable figure.

        Args:
            hook (tuple of str):
                Indicates the hook that triggered the callback. The first string will be of ``"epoch"`` or ``"step"``.
                The second string will be one of ``"train"``, ``"val"``, or ``"test"``.

            attr (any):
                The value attribute assigned to the LightningModule, retrieved with
                ``getattr(pl_module, self.attr_name)``.

            trainer (:class:`pytorch_lightning.Trainer`)
                Pytorch Lightning trainer instance

            pl_module (:class:`pytorch_lightning.LightningModule`)
                LightningModule instance

            step (int):
                Current step associated with the triggering callback event. When ``epoch_counter=True`` this will
                be in units of epochs, otherwise it will be in units of global steps.

        Example:
            >>> def prepare_figure(self, hook, attr, trainer, pl_module, step):
            >>>     'Converts attribute from tuple of tensors into a loggable scatterplot'
            >>>     x, y = attr
            >>>     fig = plt.figure()
            >>>     fig.scatter(x, y)
            >>>     setattr(pl_module, self.attr_name, fig)
        """

    def callback_fn(
        self,
        hook: Tuple[str, str],
        attr: Any,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        step: int,
        batch_idx: Optional[int],
    ) -> None:
        _hook, _mode = hook

        self.prepare_figure(hook, attr, trainer, pl_module, step)
        if isinstance(attr, plt.Figure):
            name = f"{_mode}/{self.name}"
            for f in self.log_fn:
                f(name, attr, trainer, pl_module, step, batch_idx, self)
            if self.close:
                plt.close(fig=attr)

        elif not self.ignore_errors:
            raise TypeError(f"Expected {self.attr_name} to be a Pyplot figure, but found {type(attr)}")
