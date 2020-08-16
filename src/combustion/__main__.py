#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
import sys
from typing import Any, Callable, Optional, Tuple

import hydra
import hydra.experimental
import pytorch_lightning as pl
from hydra.core.global_hydra import GlobalHydra
from hydra.types import RunMode
from omegaconf import DictConfig
from packaging import version

import combustion
from combustion.lightning import HydraMixin


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
        log.warn("One or more runs raised an exception")
        raise MultiRunError(f"Exceptions: {_exceptions}")


def clear_exceptions():
    global _exceptions
    _exceptions = []


def auto_lr_find(cfg: DictConfig, model: pl.LightningModule) -> Optional[float]:
    r"""Performs automatic learning rate selection using PyTorch Lightning.
    This is essentially a wrapper function that invokes PyTorch Lightning's
    auto LR selection using Hydra inputs. The model's learning rate is
    automatically set to the selected learning rate, and the selected
    learning rate is logged. If possible, a plot of the learning rate
    selection curve will also be produced.

    Args:

        cfg (DictConfig):
            The Hydra config

        model (LightningModule):
            The model to select a learning rate for.

    Returns:
        The learning rate if one was found, otherwise ``None``.
    """
    # store original precision, set trainer to 32 bit mode for stability
    if "precision" in cfg.trainer["params"].keys():
        precision = cfg.trainer["params"]["precision"]
        cfg.trainer["params"]["precision"] = 32
    else:
        precision = 32

    lr = None
    try:
        model.prepare_data()
        lr_trainer: pl.Trainer = HydraMixin.instantiate(cfg.trainer)
        lr_finder = lr_trainer.lr_find(model)
        lr = lr_finder.suggestion()
        log.info("Found learning rate %f", lr)
        cfg.optimizer["params"]["lr"] = lr

        # save lr curve figure
        try:
            cwd = os.getcwd()
            path = os.path.join(cwd, "lr_curve.png")
            fig = lr_finder.plot(suggest=True)
            log.info("Saving LR curve to %s", path)
            fig.savefig(path)
            fig.close()
        except Exception as err:
            log.exception(err)
            log.info("No learning rate curve was saved")

    except Exception as err:
        log.exception(err)
        log.info("Learning rate auto-find failed, using learning rate specified in config")
        _exceptions.append(err)
    finally:
        if "precision" in cfg.trainer["params"].keys():
            cfg.trainer["params"]["precision"] = precision
        cfg.trainer["params"]["auto_lr_find"] = False

    return lr


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
        key, value = x.split("=")
        overrides_dict[key] = value

    # use compose api to inspect multirun values
    with hydra.experimental.initialize(config_path, caller_stack_depth=caller_stack_depth):
        assert gh.hydra is not None
        cfg = gh.hydra.compose_config(config_name=config_name, overrides=overrides, run_mode=RunMode.MULTIRUN,)
        assert isinstance(cfg, DictConfig)

        if "sweeper" in cfg.keys() and cfg.sweeper:
            log.debug("Using sweeper values: %s", cfg.sweeper)
            overrides_dict.update(cfg.sweeper)

            if "--multirun" not in flags and "-m" not in flags:
                log.warn("Multirun flag not given but sweeper config was non-empty. " "Adding -m flag")
                flags.append("--multirun")
        else:
            log.debug("No sweeper config specified")

    # append key value pairs in sweeper config to sys.argv
    overrides = [f"{key}={value}" for key, value in overrides_dict.items()]
    sys.argv = [sys.argv[0],] + flags + overrides


# accepts options from the yaml config file
# see hydra docs: https://hydra.cc/docs/intro
def main(cfg: DictConfig, process_results_fn: Optional[Callable[[Tuple[Any, Any]], Any]] = None) -> None:
    r"""Main method for training/testing of a model using PyTorch Lightning and Hydra.

    This method is robust to exceptions (other than :class:`SystemExit` or :class:`KeyboardInterrupt`),
    making it useful when using Hydra's multirun feature. If one combination of hyperparameters results in
    an exception, other combinations will still be attempted. This behavior can be overriden by providing
    a ``check_exceptions`` bool value under ``config.trainer``. Such an override is useful when writing tests.

    Automatic learning rate selection is handled automatically using :func:`auto_lr_find`.

    Training / testing is automatically performed based on the configuration keys present in ``config.dataset``.

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
    """
    train_results, test_results = None, None
    model: Optional[pl.LightningModule] = None
    trainer: Optional[pl.Trainer] = None

    try:
        _log_versions()
        log.info("Configuration: \n%s", cfg.pretty())

        if "deterministic" in cfg.trainer.params.keys():
            seed_val = 42
            log.info("Determinstic training requested, seeding everything with %d", seed_val)
            pl.seed_everything(seed_val)

        # instantiate model (and optimizer) selected in yaml
        # see pytorch lightning docs: https://pytorch-lightning.rtfd.io/en/latest
        model: pl.LightningModule = HydraMixin.instantiate(cfg.model, cfg)

        # run auto learning rate find if requested
        if "auto_lr_find" in cfg.trainer["params"]:
            if "fast_dev_run" in cfg.trainer["params"] and cfg.trainer["params"]["fast_dev_run"]:
                log.info("Skipping auto learning rate find when fast_dev_run is set")
            else:
                auto_lr_find(cfg, model)

        # instantiate trainer with params as selected in yaml
        # handles tensorboard, checkpointing, etc
        trainer: pl.Trainer = HydraMixin.instantiate(cfg.trainer)

        # train
        if "train" in cfg.dataset:
            log.info("Starting training")
            train_results = trainer.fit(model)
            log.info("Train results: %s", train_results)
        else:
            log.info("No training dataset given")

        # test
        if "test" in cfg.dataset:
            log.info("Starting testing")
            test_results = trainer.test(model)
            log.info("Test results: %s", test_results)
        else:
            log.info("No test dataset given")

        log.info("Finished!")

    # guard to continue when using Hydra multiruns
    # SystemExit/KeyboardInterrupt are not caught and will trigger shutdown
    except Exception as err:
        catch_exceptions = cfg.trainer.get("catch_exceptions", True)
        if catch_exceptions:
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
        output = process_results_fn(train_results, test_results)
    else:
        output = train_results, test_results

    return output


if __name__ == "__main__":
    _main = hydra.main(config_path="./conf/config.yaml")(main)
    _main()
