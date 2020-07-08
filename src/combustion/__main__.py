#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
from typing import Optional

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig

from combustion.lightning import HydraMixin


log = logging.getLogger(__name__)

_exceptions = []


class MultiRunError(RuntimeError):
    pass


def _log_versions():
    import torch

    log.info("Versions:")
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


# accepts options from the yaml config file
# see hydra docs: https://hydra.cc/docs/intro
def main(cfg: DictConfig) -> None:
    r"""Main method for training/testing of a model using PyTorch Lightning and Hydra.

    This method is robust to exceptions (other than :class:`SystemExit` or :class:`KeyboardInterrupt`),
    making it useful when using Hydra's multirun feature. If one combination of hyperparameters results in
    an exception, other combinations will still be attempted.

    Automatic learning rate selection is handled automatically using :func:`auto_lr_find`.

    Training / testing is automatically performed based on the configuration keys present in ``config.dataset``.

    Args:

        cfg (DictConfig):
            The Hydra config

    Example::

        # define main method as per Hydra that calls combustion.main()
        @hydra.main(config_path="./conf", config_name="config")
        def main(cfg):
            combustion.main(cfg)

        if __name__ == "__main__":
            main()
    """
    try:
        _log_versions()
        log.info("Configuration: \n%s", cfg.pretty())

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
            trainer.fit(model)
        else:
            log.info("No training dataset given")

        # test
        if "test" in cfg.dataset:
            log.info("Starting testing")
            trainer.test(model)
        else:
            log.info("No test dataset given")

        log.info("Finished!")

    # guard to continue when using Hydra multiruns
    # SystemExit/KeyboardInterrupt are not caught and will trigger shutdown
    except Exception as err:
        log.exception(err)
        _exceptions.append(err)


if __name__ == "__main__":
    _main = hydra.main(config_path="./conf/config.yaml")(main)
    _main()
