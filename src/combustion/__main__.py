#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig

from combustion.lightning import HydraMixin


log = logging.getLogger(__name__)

_exceptions = []


def check_exceptions():
    for x in _exceptions:
        raise x


def auto_lr_find(cfg: DictConfig, model: pl.LightningModule) -> None:
    # store original precision, set trainer to 32 bit mode for stability
    if "precision" in cfg.trainer["params"].keys():
        precision = cfg.trainer["params"]["precision"]
        cfg.trainer["params"]["precision"] = 32
    else:
        precision = 32

    try:
        model.prepare_data()
        lr_trainer: pl.Trainer = HydraMixin.instantiate(cfg.trainer)
        lr_finder = lr_trainer.lr_find(model)
        log.info("Found learning rate %f", lr_finder.suggestion())
        cfg.optimizer["params"]["lr"] = lr_finder.suggestion()

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


# accepts options from the yaml config file
# see hydra docs: https://hydra.cc/docs/intro
def main(cfg: DictConfig, train=False, test=False):
    try:
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
