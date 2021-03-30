#!/usr/bin/env python
# -*- coding: utf-8 -*-


from pathlib import Path
from typing import Any, Iterable, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
from torch import Tensor

from .base import AttributeCallback


class SaveTensors(AttributeCallback):
    """Callback to save arbitrary tensors to disk.

    Example:
        >>> callback = SaveTensors()
        >>>
        >>> # LightningModule.training_step
        >>> def training_step(self, batch, batch_idx):
        >>>     image, target = batch
        >>>     ...
        >>>     # attribute will be saved under
        >>>     # {trainer.default_root_dir}/{mode}/last_image_{step}.pth
        >>>     self.last_attr = attr

    Args:
        path (str, optional):
            Path where tensors will be saved. By default, the trainer's ``default_root_dir``
            will be used.

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
        path: Optional[str] = None,
        triggers: Union[str, Iterable[str]] = ("train", "val", "test"),
        hook: str = "step",
        attr_name: str = "last_attr",
        epoch_counter: bool = False,
        max_calls: Optional[int] = None,
        interval: Optional[int] = None,
        ignore_errors: bool = False,
        **kwargs,
    ):
        super().__init__(triggers, hook, attr_name, epoch_counter, max_calls, interval, ignore_errors)
        self.path = Path(path) if path is not None else None
        self.kwargs = kwargs

    def callback_fn(
        self, hook: Tuple[str, str], attr: Any, trainer: pl.Trainer, pl_module: pl.LightningModule, step: int
    ) -> None:
        _hook, _mode = hook
        if not isinstance(attr, Tensor):
            if self.ignore_errors:
                return
            else:
                raise TypeError(f"Expected type({self.attr_name}) == Tensor, found {type(attr)}")

        base_path = self.path if self.path is not None else trainer.default_root_dir
        path = Path(base_path, _mode, f"{self.attr_name}_{step}.pth")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(attr, path, **self.kwargs)
        except RuntimeError:
            if not self.ignore_errors:
                raise
