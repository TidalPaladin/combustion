#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABC
from copy import deepcopy
from typing import Any, Optional, Union

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset


class HydraMixin(ABC):
    r"""
    Mixin for creating LightningModules using Hydra
    """

    def get_lr(self, pos: int = 0, param_group: int = 0) -> float:
        if not self.trainer.lr_schedulers:
            return self.trainer.optimizer.state_dict()["param_groups"][param_group]["lr"]
        else:
            scheduler = self.trainer.lr_schedulers[pos]["scheduler"]
            return scheduler.get_last_lr()[param_group]

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

    def prepare_data(self) -> None:
        self.train_ds: Optional[Dataset] = self._prepare_data("train")
        self.val_ds: Optional[Dataset] = self._prepare_data("validate")
        self.test_ds: Optional[Dataset] = self._prepare_data("test")

        num_workers = self.config.dataset.get("num_workers")
        self.num_workers = num_workers if num_workers is not None else 1

    def train_dataloader(self) -> Optional[DataLoader]:
        return self._dataloader(self.train_ds)

    def val_dataloader(self) -> Optional[DataLoader]:
        return self._dataloader(self.val_ds)

    def test_dataloader(self) -> Optional[DataLoader]:
        return self._dataloader(self.test_ds)

    def _prepare_data(self, subset: str) -> Optional[Dataset]:
        if subset in self.config.dataset.keys():
            return HydraMixin.instantiate(self.config.dataset[subset])
        else:
            return None

    def _dataloader(self, dataset: Optional[Dataset]) -> Optional[DataLoader]:
        if dataset is not None:
            return DataLoader(dataset, num_workers=self.num_workers, batch_size=self.hparams["batch_size"],)
        else:
            return None

    @staticmethod
    def instantiate(config: Union[DictConfig, dict], *args, **kwargs) -> Any:
        r"""
        Recursively instantiates classes in a Hydra configuration.
        """
        # deepcopy so we can modify config
        config = deepcopy(config)
        if isinstance(config, DictConfig):
            OmegaConf.set_struct(config, False)
        params = dict(config.get("params")) if "params" in config.keys() else {}

        def is_subclass(d):
            return isinstance(d, (dict, DictConfig)) and "cls" in d.keys()

        subclasses = {key: subconfig for key, subconfig in params.items() if is_subclass(subconfig)}

        # instantiate recursively, remove those keys from config used in hydra instantiate call
        for key, subconfig in subclasses.items():
            subclasses[key] = HydraMixin.instantiate(subconfig)
            del config.get("params")[key]

        subclasses.update(kwargs)
        return instantiate(config, *args, **subclasses)
