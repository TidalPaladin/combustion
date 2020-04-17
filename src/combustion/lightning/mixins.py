#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABC
from copy import deepcopy
from typing import Any, Union

from hydra.utils import instantiate
from omegaconf import DictConfig


class HydraMixin(ABC):
    r"""
    Mixin for creating LightningModules using Hydra
    """

    def get_lr(self, pos: int = 0, param_group: int = 0) -> float:
        if not self.trainer.lr_schedulers:
            return self.trainer.optimizer.state_dict()["param_groups"][param_group]["lr"]
        else:
            scheduler = self.trainer.lr_schedulers[pos]["scheduler"]
            return scheduler.get_lr()[param_group]

    def configure_optimizers(self):
        if not hasattr(self, "config"):
            raise AttributeError("'config' attribute is required for configure_optimizers")

        lr = self.config.optimizer["params"]["lr"]
        optim = instantiate(self.config.optimizer, self.parameters())

        schedule = self.config.get("schedule")
        if schedule is not None:
            steps_per_epoch = len(self.train_dataloader())
            schedule_dict = {
                "interval": schedule.get("interval", "epoch"),
                "monitor": schedule.get("monitor", "val_loss"),
                "frequency": schedule.get("frequency", 1),
                "scheduler": instantiate(schedule, optim, max_lr=lr, steps_per_epoch=steps_per_epoch),
            }
            return [optim], [schedule_dict]

        return optim

    @staticmethod
    def instantiate(config: Union[DictConfig, dict], *args, **kwargs) -> Any:
        r"""
        Recursively instantiates classes in a Hydra configuration.
        """
        # deepcopy so we can modify config
        config = deepcopy(config)
        params = dict(config.get("params")) if "params" in config.keys() else {}

        def is_subclass(d):
            if not isinstance(d, (dict, DictConfig)):
                return False
            if not any([k in d.keys() for k in ("class", "cls")]):
                return False
            return True

        subclasses = {key: subconfig for key, subconfig in params.items() if is_subclass(subconfig)}

        # instantiate recursively, remove those keys from config used in hydra instantiate call
        for key, subconfig in subclasses.items():
            subclasses[key] = HydraMixin.instantiate(subconfig)
            del config.get("params")[key]

        subclasses.update(kwargs)
        return instantiate(config, *args, **subclasses)
