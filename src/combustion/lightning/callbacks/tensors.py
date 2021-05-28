#!/usr/bin/env python
# -*- coding: utf-8 -*-


import csv
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
from torch import Tensor

from .base import AttributeCallback, mkdir, resolve_dir


class SaveTensors(AttributeCallback):
    """Callback to save arbitrary tensors to disk. Tensors can be saved in PyTorch,
    Matlab, or CSV format.

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

        output_format (str):
            Format for saving tensors. Can be a single string or iterable of strings.

            Each string should be one of:

                * ``"pth"`` - Saved with :func:`torch.save`
                * ``"mat"`` - Matlab format with :func:`scipy.io.savemat`
                * ``"csv"`` - CSV file (for 2D or 1D tensors only)
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
        output_format: str = "pth",
        **kwargs,
    ):
        super().__init__(triggers, hook, attr_name, epoch_counter, max_calls, interval, ignore_errors)
        self.path = Path(path) if path is not None else None
        self.kwargs = kwargs

        VALID_FORMATS = ("pth", "mat", "csv")
        if isinstance(output_format, str):
            self.output_format = str(output_format).lower()
            if self.output_format not in VALID_FORMATS:
                raise ValueError(f"Invalid `output_format` {self.output_format}")
        else:
            self.output_format = tuple(str(x).lower() for x in output_format)
            for x in self.output_format:
                if x not in VALID_FORMATS:
                    raise ValueError(f"Invalid `output_format` {x}")

    @staticmethod
    def save_tensor(p: Path, t: Tensor, output_format: str, **kwargs) -> None:
        if output_format == "pth":
            torch.save(t, p, **kwargs)

        elif output_format == "mat":
            try:
                from scipy.io import savemat
            except ModuleNotFoundError:
                print("Saving tensors in Matlab format requires scipy")
                raise
            # NOTE: must explicitly cast to numpy or you get empty files
            savemat(p, {"tensor": t.cpu().numpy()}, **kwargs)

        elif output_format == "csv":
            if t.ndim > 2:
                raise ValueError(f"Saving tensors CSV requires tensor.ndim <= 2, found {t.ndim}")
            with open(p, "w", newline="") as csvfile:
                writer = csv.writer(csvfile, delimiter=" ", **kwargs)
                for row in t:
                    writer.writerow(row.tolist())

        else:
            raise ValueError(f"Unknown `output_format` {output_format}")

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
        if not isinstance(attr, Tensor):
            if self.ignore_errors:
                return
            else:
                raise TypeError(f"Expected type({self.attr_name}) == Tensor, found {type(attr)}")

        base_path = self.path
        path = Path(base_path, _mode, f"{self.attr_name}", self.read_step_as_str(pl_module, batch_idx)).with_suffix(
            ".pth"
        )
        output_format = (self.output_format,) if isinstance(self.output_format, str) else self.output_format
        try:
            mkdir(path.parent, trainer)
            for f in output_format:
                self.save_tensor(path.with_suffix(f".{f}"), attr, f, **self.kwargs)

        except RuntimeError:
            if not self.ignore_errors:
                raise

    def on_pretrain_routine_start(self, trainer, pl_module):
        """When pretrain routine starts we build the dest dir on the fly"""
        path = resolve_dir(trainer, self.path, "saved_tensors")
        self.path = Path(path)

    def on_test_start(self, trainer, pl_module):
        """When pretrain routine starts we build the dest dir on the fly"""
        self.on_pretrain_routine_start(trainer, pl_module)

    def on_predict_start(self, trainer, pl_module):
        """When pretrain routine starts we build the dest dir on the fly"""
        self.on_pretrain_routine_start(trainer, pl_module)
