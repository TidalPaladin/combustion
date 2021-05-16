#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import re
import sys
from typing import Any, Callable, Optional, Tuple

import hydra
import hydra.experimental
import pytorch_lightning as pl
from hydra.core.global_hydra import GlobalHydra
from hydra.types import RunMode
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from packaging import version

import combustion

from .hydra_conf import CombustionConf


log = logging.getLogger("combustion")

_exceptions = []


class MultiRunError(RuntimeError):
    pass


def _log_versions():
    import torch

    log.info("Versions:")
    log.info("\tcombustion: %s", combustion.__version__)
    log.info("\ttorch: %s", torch.__version__)
    log.info("\tpytorch_lightning: %s", pl.__version__)
    log.info("\thydra: %s", hydra.__version__)
    try:
        import torchvision

        log.info("\ttorchvision: %s", torchvision.__version__)
    except ImportError:
        pass
    try:
        import kornia

        log.info("\tkornia: %s", kornia.__version__)
    except ImportError:
        pass


def check_exceptions():
    r"""Checks if exceptions have been raised over the course of a multirun.
    Most exceptions are ignored by :func:`combustion.main` to prevent a failed run
    from killing an entire hyperparameter search. However, one may still want to
    raise an exception at the conclusion of a multirun (i.e. for testing purposes).
    This method checks if any exceptions were raised, and if so will raise a
    :class:`combustion.MultiRunError`.

    Example::

        >>> @hydra.main(config_path="./conf", config_name="config")
        >>> def main(cfg):
        >>>     combustion.main(cfg)
        >>>
        >>> if __name__ == "__main__":
        >>>     main()
        >>>     combustion.check_exceptions()

    """
    global _exceptions
    if _exceptions:
        log.warning("One or more runs raised an exception")
        raise MultiRunError(f"Exceptions: {_exceptions}")


def clear_exceptions():
    global _exceptions
    _exceptions = []


def initialize(config_path: str, config_name: str, caller_stack_depth: int = 1) -> None:
    r"""Performs initialization needed for configuring multiruns / parameter sweeps via
    a YAML config file. Currently this is only needed if multirun configuration via a YAML
    file is desired. Otherwise, :func:`combustion.main` can be used without calling ``initialize``.
    See :func:`combustion.main` for usage examples.

    .. warning::
        This method makes use of Hydra's compose API, which is experimental as of version 1.0.

    .. warning::
        This method works by inspecting the "sweeper" section of the specified config file
        and altering ``sys.argv`` to include the chosen sweeper parameters.

    Args:
        config_path (str):
            Path to main configuration file. See :func:`hydra.main` for more details.

        config_name (str):
            Name of the main configuration file. See :func:`hydra.main` for more details.

        caller_stack_depth (int):
            Stack depth when calling :func:`initialize`. Defaults to 1 (direct caller).


    Sample sweeper Hydra config
        .. code-block:: yaml

            sweeper:
              model.params.batch_size: 8,16,32
              optimizer.params.lr: 0.001,0.002,0.0003
    """
    assert caller_stack_depth >= 1
    caller_stack_depth += 1

    if version.parse(hydra.__version__) < version.parse("1.0.0rc2"):
        raise ImportError(f"Sweeping requires hydra>=1.0.0rc2, but you have {hydra.__version__}")

    gh = GlobalHydra.instance()
    if GlobalHydra().is_initialized():
        gh.clear()

    flags = [x for x in sys.argv[1:] if x[0] == "-"]
    overrides = [x for x in sys.argv[1:] if x[0] != "-"]

    # split argv into dict
    overrides_dict = {}
    for x in overrides:
        key, value = re.split(r"=|\s", x, maxsplit=1)
        overrides_dict[key] = value

    # use compose api to inspect multirun values
    with hydra.experimental.initialize(config_path, caller_stack_depth=caller_stack_depth):
        assert gh.hydra is not None
        cfg = gh.hydra.compose_config(
            config_name=config_name,
            overrides=overrides,
            run_mode=RunMode.MULTIRUN,
        )
        assert isinstance(cfg, DictConfig)

        if "sweeper" in cfg.keys() and cfg.sweeper:
            log.debug("Using sweeper values: %s", cfg.sweeper)
            overrides_dict.update(cfg.sweeper)

            if "--multirun" not in flags and "-m" not in flags:
                log.warning("Multirun flag not given but sweeper config was non-empty. " "Adding -m flag")
                flags.append("--multirun")
        else:
            log.debug("No sweeper config specified")

    # append key value pairs in sweeper config to sys.argv
    overrides = [f"{key}={value}" for key, value in overrides_dict.items()]
    sys.argv = (
        [
            sys.argv[0],
        ]
        + flags
        + overrides
    )


# accepts options from the yaml config file
# see hydra docs: https://hydra.cc/docs/intro
def main(cfg: CombustionConf, process_results_fn: Optional[Callable[[Tuple[Any, Any]], Any]] = None) -> None:
    r"""Main method for training/testing of a model using PyTorch Lightning and Hydra.

    This method is robust to exceptions (other than :class:`SystemExit` or :class:`KeyboardInterrupt`),
    making it useful when using Hydra's multirun feature. If one combination of hyperparameters results in
    an exception, other combinations will still be attempted. This behavior can be overriden by providing
    a ``check_exceptions`` bool value under ``config.trainer``. Such an override is useful when writing tests.

    Automatic learning rate selection is handled using :func:`auto_lr_find`.

    Training / testing is automatically performed based on the configuration keys present in ``config.dataset``.

    Additionally, the following Hydra overrides are supported:
        * ``trainer.load_from_checkpoint`` - Load model weights (but not training state) from a checkpoint
        * ``trainer.test_only`` - Skip training even if a training dataset is given

    Args:

        cfg (DictConfig):
            The Hydra config

        process_results_fn (callable, optional):
            If given, call ``process_results_fn`` on the ``(train_results, test_results)`` tuple returned by
            this method. This is useful for processing training/testing results into a scalar return value
            when using an optimization sweeper (like Ax).

    Example::

        >>> # define main method as per Hydra that calls combustion.main()
        >>> @hydra.main(config_path="./conf", config_name="config")
        >>> def main(cfg):
        >>>     combustion.main(cfg)
        >>>
        >>> if __name__ == "__main__":
        >>>     main()

    Example (multirun from config file)::

        >>> combustion.initialize(config_path="./conf", config_name="config")
        >>>
        >>> @hydra.main(config_path="./conf", config_name="config")
        >>> def main(cfg):
        >>>     return combustion.main(cfg)
        >>>
        >>> if __name__ == "__main__":
        >>>     main()
        >>>     combustion.check_exceptions()

    Example (inference-time command)::
        ``python -m my_module trainer.load_from_checkpoint=foo.ckpt trainer.test_only=True``
    """
    trainer: Optional[pl.Trainer] = None
    try:
        _log_versions()
        log.info("Configuration: \n%s", OmegaConf.to_yaml(cfg))
        pl.seed_everything(cfg.seed)

        trainer: pl.Trainer = instantiate(cfg.trainer)
        data: pl.LightningDataModule = instantiate(cfg.data, _recursive_=False)
        assert isinstance(trainer, pl.Trainer)
        assert isinstance(data, pl.LightningDataModule)

        model: pl.LightningModule = instantiate(cfg.model, _recursive_=False)
        assert isinstance(model, pl.LightningModule)
        if cfg.load_checkpoint:
            model = model.__class__.load_from_checkpoint(cfg.checkpoint)

        results: Dict[str, Any] = {}
        if cfg.fit:
            log.info("Starting training")
            trainer.tune(model, datamodule=data)
            results["train"] = trainer.fit(model, datamodule=data)
            log.info("Train results: %s", results["train"])

        if cfg.test:
            log.info("Starting testing")
            results["test"] = trainer.test(model, datamodule=data)
            log.info("Test results: %s", results["test"])

        if cfg.predict:
            log.info("Making predictions")
            results["predict"] = trainer.predict(model, datamodule=data)

        log.info("Finished!")

    # guard to continue when using Hydra multiruns
    # SystemExit/KeyboardInterrupt are not caught and will trigger shutdown
    except Exception as err:
        if cfg.catch_exceptions:
            log.exception(err)
            _exceptions.append(err)
        else:
            raise err

    finally:
        # flush logger to ensure free memory for next run
        if trainer is not None:
            experiment = trainer.logger.experiment
            if experiment is not None and hasattr(experiment, "flush"):
                experiment.flush()
                log.debug("Flushed experiment to disk")
            if experiment is not None and hasattr(experiment, "close"):
                experiment.close()
                log.debug("Closed experiment writer")

    # postprocess results if desired (e.g. to scalars for bayesian optimization)
    if process_results_fn is not None:
        log.debug("Running results postprocessing")
        results = process_results_fn(results)

    return results


if __name__ == "__main__":
    _main = hydra.main(config_path="./conf/config.yaml")(main)
    _main()
