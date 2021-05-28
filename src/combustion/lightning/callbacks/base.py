#!/usr/bin/env python
# -*- coding: utf-8 -*-

import inspect
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback


def mkdir(path, trainer):
    r"""Creates a directory, accounting for multi-process trainers"""
    if not trainer.fast_dev_run and trainer.is_global_zero:
        Path(path).mkdir(exist_ok=True, parents=True)


def resolve_dir(trainer: pl.Trainer, dirpath: Optional[str], suffix: str):
    """
    Determines tensor save directory at runtime. References attributes from the
    trainer's logger to determine where to save checkpoints.
    The base path for saving weights is set in this priority:
    1.  ``dirpath`` if it is passed in
    2.  The default_root_dir from trainer if trainer has no logger
    The base path gets extended with logger name and version (if these are available)
    and subfolder ``suffix``.
    """
    if dirpath is not None:
        mkdir(dirpath, trainer)
        return dirpath
    else:
        path = Path(trainer.log_dir, suffix)

    mkdir(path, trainer)
    return path


class AttributeCallback(Callback, ABC):
    """Base class for callbacks that read an attribute assigned to the LightningModule
    and perform some logic on batch or epoch end.

    Args:
        triggers (str or list of str):
            Modes for which the callback should run. Must be one of
            ``"train"``, ``"val"``, ``"test"``.

        hook (str):
            One of ``"step"`` or ``"epoch"``, determining if the callback will be triggered on
            batch end or epoch end.

        attr_name (str):
            Name of the attribute where the callback will search for the image to be logged.

        epoch_counter (bool):
            If ``True``, report the epoch for each callback invocation. By default, the
            global step is reported.

        max_calls (int, optional):
            If given, do not trigger the callback more than ``max_calls`` times per epoch.

        interval (int, optional):
            If given, only execute the callback every ``interval`` steps
            (or epochs when ``epoch_counter=True``)

        ignore_errors (bool):
            If ``True``, do not raise an exception if ``attr_name`` cannot be found.
    """

    def __init__(
        self,
        triggers: Union[str, Iterable[str]] = ("train", "val", "test"),
        hook: str = "step",
        attr_name: str = "last_attr",
        epoch_counter: bool = False,
        max_calls: Optional[int] = None,
        interval: Optional[int] = None,
        ignore_errors: bool = False,
    ):
        self.triggers = (
            tuple(str(x).lower() for x in triggers) if isinstance(triggers, Iterable) else (str(triggers).lower(),)
        )
        self.hook = str(hook)
        self.attr_name = str(attr_name)
        self.max_calls = int(max_calls) if max_calls is not None else None
        self.ignore_errors = bool(ignore_errors)
        self.interval = int(interval) if interval is not None else None
        self.epoch_counter = bool(epoch_counter)
        self.counter = 0

        if self.hook not in ("step", "epoch"):
            raise ValueError(f"Expected `hook` to be 'epoch' or 'step', found {hook}")
        for t in triggers:
            if t not in ("train", "val", "test"):
                raise ValueError(f"Invalid trigger {t}, triggers must be one of 'train', 'val', 'test'")
        if not isinstance(epoch_counter, bool):
            raise TypeError(f"Expected type(epoch_counter) == bool, found {type(epoch_counter)}")
        for name, var in zip(("interval", "max_calls"), (self.interval, self.max_calls)):
            if var is not None and var <= 0:
                raise ValueError(f"{name} must be > 0, found {var}")

    @abstractmethod
    def callback_fn(
        self,
        hook: Tuple[str, str],
        attr: Any,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        step: int,
        batch_idx: Optional[int],
    ) -> None:
        r"""Function to be implemented containing the callback instructions.

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

            batch_idx (optional int):
                Batch index associated with the triggering callback event. Only available for ``"step"`` hooks.
        """
        raise NotImplementedError("callback_fn")

    def has_attribute(self, pl_module: pl.LightningModule) -> bool:
        r"""Checks if the target attribute exists on a LightningModule.

        Args:
            pl_module (:class:`pytorch_lightning.LightningModule`):
                Module to search for ``self.attr_name``
        """
        return hasattr(pl_module, self.attr_name)

    def read_attribute(self, pl_module: pl.LightningModule) -> Any:
        r"""Attempts to read the target attribute from a LightningModule.

        Args:
            pl_module (:class:`pytorch_lightning.LightningModule`):
                Module to read ``self.attr_name`` from

        Raises:
            :class:`AttributeError` if ``self.ignore_errors=False`` and the attribute was not found

        Returns:
            ``getattr(pl_module, self.attr_name)`` or ``None`` if ``self.ignore_errors=True`` and the
            attribute was missing.
        """
        if not self.has_attribute(pl_module):
            if self.ignore_errors:
                return None
            else:
                raise AttributeError(f"Module missing expected attribute {self.attr_name}")
        return getattr(pl_module, self.attr_name)

    def read_step(self, pl_module: pl.LightningModule) -> int:
        r"""Attempts to read the callback step value from a LightningModule.

        Args:
            pl_module (:class:`pytorch_lightning.LightningModule`):
                Module to read from

        Returns:
            Current step associated with the module. When ``epoch_counter=True`` this will
            be in units of epochs, otherwise it will be in units of global steps.
        """
        return pl_module.current_epoch if self.epoch_counter else pl_module.global_step

    def read_step_as_str(self, pl_module: pl.LightningModule, batch_idx: Optional[int] = None) -> str:
        r"""Attempts to read the callback step value from a LightningModule as a string

        Args:
            pl_module (:class:`pytorch_lightning.LightningModule`):
                Module to read from

            batch_idx (int):
                Optional batch index

        Returns:
            Current step associated with the module. When ``epoch_counter=True`` this will
            be in units of epochs, otherwise it will be in units of global steps. The string
            will be prefixed with ``"epoch"`` or ``"step"`` depending on ``epoch_counter``.
        """
        step = self.read_step(pl_module)
        if self.epoch_counter:
            return f"epoch_{step}"
        elif batch_idx is not None:
            return f"step_{step}_batch_{batch_idx}"
        else:
            return f"step_{step}"

    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args) -> None:
        return self._on_hook_end(("step", "train"), trainer, pl_module, *args)

    def on_validation_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args) -> None:
        return self._on_hook_end(("step", "val"), trainer, pl_module, *args)

    def on_test_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args) -> None:
        return self._on_hook_end(("step", "test"), trainer, pl_module, *args)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args) -> None:
        return self._on_hook_end(("epoch", "train"), trainer, pl_module, *args)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args) -> None:
        return self._on_hook_end(("epoch", "val"), trainer, pl_module, *args)

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args) -> None:
        return self._on_hook_end(("epoch", "test"), trainer, pl_module, *args)

    def _on_hook_end(self, hook: Tuple[str, str], trainer: pl.Trainer, pl_module: pl.LightningModule, *args) -> None:
        _hook, _mode = hook
        if self.hook != _hook or _mode not in self.triggers:
            return
        with torch.no_grad():
            self._hook(hook, trainer, pl_module, *args)

    def _hook(self, hook: Tuple[str, str], trainer: pl.Trainer, pl_module: pl.LightningModule, *args) -> None:
        _hook, _mode = hook
        if _mode not in self.triggers:
            return

        attr = self.read_attribute(pl_module)
        if attr is None:
            return
        step = self.read_step(pl_module)

        # skip if enough images have already been logged
        if self.max_calls is not None and self.counter >= self.max_calls:
            return
        # skip if logging is desired at a non-unit interval
        elif self.interval is not None and step % self.interval != 0:
            return

        get_index = 2
        batch_idx = args[get_index] if len(args) >= get_index + 1 else None
        self.callback_fn(hook, attr, trainer, pl_module, step, batch_idx)
        self.counter += 1

    def on_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if hasattr(pl_module, self.attr_name):
            delattr(pl_module, self.attr_name)
        self.counter = 0

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}("
        init_signature = inspect.signature(self.__class__)
        defaults = {
            k: v.default if v.default is not inspect.Parameter.empty else None
            for k, v in init_signature.parameters.items()
        }

        for i, (name, default) in enumerate(defaults.items()):
            val = getattr(self, name)
            display_val = f"'{val}'" if isinstance(val, str) else val
            if name != "kwargs" and val != default:
                s += f"{',' if i == 0 else ''} {name}={display_val}"
        s += ")"
        return s
