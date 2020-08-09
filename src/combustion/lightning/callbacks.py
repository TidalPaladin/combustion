#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import warnings
from typing import Any, Callable, Dict, Optional, Tuple

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks.base import Callback
from torch.jit import ScriptModule


try:
    from thop import profile
except ImportError:

    def profile(*args, **kwargs):
        raise ImportError(
            "CountMACs requires thop. "
            "Please install combustion with 'macs' extras using "
            "pip install combustion [macs]"
        )


log = logging.getLogger(__name__)


class TorchScriptCallback(Callback):
    r"""Callback to export a model using TorchScript upon completion of training.

    .. note::

        A type hint of :class:`pytorch_lightning.LightningModule`, ``_device: ...`` causes
        problems with TorchScript exports. This type hint must be manually overridden
        as follows::

            >>> class MyModule(pl.LightningModule):
            >>>     _device: torch.device
            >>>     ...

    Args:
        path (str, optional):
            The filepath where the exported model will be saved. If unset, the model will be saved
            in the PyTorch Lightning default save path.

        trace (bool, optional):
            If true, export a :class:`torch.jit.ScriptModule` using :func:`torch.jit.trace`.
            Otherwise, :func:`torch.jit.script` will be used.

        sample_input (Any, optional):
            Sample input data to use with :func:`torch.jit.trace`. If ``sample_input`` is unset and
            ``trace`` is true, the attribute :attr:`example_input_array` will be used as input. If
            ``trace`` is true and :attr:`example_input_array` is unset a :class:`RuntimeError` will
            be raised.
    """

    def __init__(self, path: Optional[str] = None, trace: bool = False, sample_input: Optional[Any] = None):
        self.path = path
        self.trace = trace
        self.sample_input = sample_input

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        r"""Called after training to export a model using TorchScript.

        Args:
            trainer:
                The :class:`pytorch_lightning.Trainer` instance

            pl_module:
                The :class:`pytorch_lightning.LightningModule` to export.
        """
        # check _device annotation is not ...
        # scripting will fail if _device type annotation is not overridden
        if pl_module.__annotations__["_device"] == ...:
            raise RuntimeError(
                "Please override type annotation for pl_module._device for scripting to work. "
                "Using _deivce: torch.device seems to work."
            )

        # get training state of model so it can be restored later
        training = pl_module.training
        if training:
            pl_module.eval()

        path = self.path if self.path is not None else self._get_default_save_path(trainer)

        if self.trace and self.sample_input is None:
            if not hasattr(pl_module, "example_input_array"):
                raise RuntimeError(
                    "Trace export was requested, but sample_input was not given and "
                    "module.example_input_array was not set."
                )
            self.sample_input = pl_module.example_input_array

        if self.trace:
            log.debug("Tracing %s", pl_module.__class__.__name__)
            script = self._get_trace(pl_module)
        else:
            log.debug("Scripting %s", pl_module.__class__.__name__)
            script = self._get_script(pl_module)
        torch.jit.save(script, path)
        log.info("Exported ScriptModule to %s", path)

        # restore training state
        if training:
            pl_module.train()

    def _get_trace(self, pl_module: pl.LightningModule) -> ScriptModule:
        assert self.sample_input is not None
        return torch.jit.trace(pl_module, self.sample_input)

    def _get_script(self, pl_module: pl.LightningModule) -> ScriptModule:
        return torch.jit.script(pl_module)

    def _get_default_save_path(self, trainer: pl.Trainer) -> str:
        if hasattr(trainer, "default_root_dir"):
            return trainer.default_root_dir
        # backwards compat
        elif hasattr(trainer, "default_save_path"):
            return trainer.default_save_path
        else:
            import os
            import warnings

            warnings.warn("Failed to find default path attribute on Trainer")
            return os.getcwd()


class CountMACs(Callback):
    r"""Callback to output the approximate number of MAC (multiply accumulate) operations
    and parameters in a model. Runs at start of training.

    .. note::
        Counting MACs requires `thop <https://github.com/Lyken17/pytorch-OpCounter>`_

    Total MACs / parameters are logged and attached to the model as attributes:

        * ``total_macs``
        * ``total_params``

    Args:
        sample_input (optional, Tuple):
            Sample input data to use when counting MACs. If ``sample_input`` is not given the callback
            will attempt to use attribute ``module.example_input_array`` as a sample input. If no sample
            input can be found a warning will be raised.

        custom_ops (optional, Dict[type, Callable]):
            Forwarded to :func:`htop.profile`
    """

    def __init__(self, sample_input: Optional[Tuple[Any]] = None, custom_ops: Optional[Dict[type, Callable]] = None):
        self.custom_ops = custom_ops
        self.sample_input = sample_input

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        r"""Called at start of training

        Args:
            trainer:
                The :class:`pytorch_lightning.Trainer` instance

            pl_module:
                The :class:`pytorch_lightning.LightningModule` to analyze.
        """

        if self.sample_input is None:
            if not hasattr(pl_module, "example_input_array") or pl_module.example_input_array is None:
                warnings.warn(
                    "MAC counting was requested, but no example input was provided. " "Skipping MAC counting."
                )
                return
            self.sample_input = pl_module.example_input_array

        if not isinstance(self.sample_input, tuple):
            inputs = (self.sample_input,)
        else:
            inputs = self.sample_input

        macs, params = profile(pl_module, inputs=inputs, custom_ops=self.custom_ops)
        macs, params = int(macs), int(params)
        log.info("Model MACs: %d", macs)
        log.info("Model Parameters: %d", params)

        for attr, source in (("macs_count", macs), ("param_count", params)):
            if hasattr(pl_module, attr):
                warnings.warn(f"Model already has attribute {attr}, skipping attribute attachment")
            else:
                setattr(pl_module, attr, source)
