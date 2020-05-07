#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABC
from copy import deepcopy
from typing import Any, Optional, Union

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch.utils.data import DataLoader, Dataset, random_split


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
        dataset_cfg = self.config.dataset
        if "train" not in dataset_cfg:
            raise MisconfigurationException("Missing 'train' dataset configuration")
        train_ds: Dataset = HydraMixin.instantiate(dataset_cfg["train"])

        # determine sizes validation/test sets if specified as a fraction of training set
        splits = {"train": len(train_ds), "validate": None, "test": None}
        for split in splits.keys():
            if not (split in dataset_cfg.keys() and isinstance(dataset_cfg[split], (int, float))):
                continue
            split_value = dataset_cfg[split]

            # for ints, assume value is size of the subset
            # for floats, assume value is a percentage of full dataset
            if isinstance(split_value, float):
                split_value = round(len(train_ds) * split_value)

            splits[split] = split_value
            splits["train"] -= split_value

        # create datasets
        if splits["validate"] is not None and splits["test"] is not None:
            lengths = (splits["train"], splits["validate"], splits["test"])
            train_ds, val_ds, test_ds = random_split(train_ds, lengths)
        elif splits["validate"] is not None:
            lengths = (splits["train"], splits["validate"])
            train_ds, val_ds = random_split(train_ds, lengths)
            test_ds = HydraMixin.instantiate(dataset_cfg["test"]) if "test" in dataset_cfg.keys() else None
        elif splits["test"] is not None:
            lengths = (splits["train"], splits["test"])
            train_ds, test_ds = random_split(train_ds, lengths)
            val_ds = HydraMixin.instantiate(dataset_cfg["validate"]) if "validate" in dataset_cfg.keys() else None
        else:
            val_ds = HydraMixin.instantiate(dataset_cfg["validate"]) if "validate" in dataset_cfg.keys() else None
            test_ds = HydraMixin.instantiate(dataset_cfg["test"]) if "test" in dataset_cfg.keys() else None

        self.train_ds: Dataset = train_ds
        self.val_ds: Optional[Dataset] = val_ds
        self.test_ds: Optional[Dataset] = test_ds

        num_workers = self.config.dataset.get("num_workers")
        self.num_workers = num_workers if num_workers is not None else 1

    def train_dataloader(self) -> Optional[DataLoader]:
        return self._dataloader(self.train_ds)

    def val_dataloader(self) -> Optional[DataLoader]:
        return self._dataloader(self.val_ds)

    def test_dataloader(self) -> Optional[DataLoader]:
        return self._dataloader(self.test_ds)

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
