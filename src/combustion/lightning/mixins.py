#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
from abc import ABC
from copy import deepcopy
from typing import Any, Optional, Union

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch.utils.data import DataLoader, Dataset, random_split


class HydraMixin(ABC):
    r"""
    Mixin for creating :class:`pytorch_lightning.LightningModule`
    using `Hydra <https://hydra.cc/>`_.

    The following :class:`pytorch_lightning.LightningModule` abstract methods are implemented:

        * :attr:`configure_optimizers`
        * :attr:`prepare_data`
        * :attr:`train_dataloader`
        * :attr:`val_dataloader`
        * :attr:`test_dataloader`
    """

    def get_lr(self, pos: int = 0, param_group: int = 0) -> float:
        r"""Gets the current learning rate. Useful for logging learning rate when using a learning
        rate schedule.

        Args:
            pos (int, optional):
                The index of the optimizer to retrieve a learning rate from. When using a single
                optimizer this can be omitted.

            param_group (int, optional):
                The index of the parameter group to retrieve a learning rate for. When using one
                optimizer for the entire model this can be omitted.

        """
        if not self.trainer.lr_schedulers:
            return self.trainer.optimizer.state_dict()["param_groups"][param_group]["lr"]
        else:
            scheduler = self.trainer.lr_schedulers[pos]["scheduler"]
            return scheduler.get_last_lr()[param_group]

    def configure_optimizers(self):
        r"""Override for :class:`pytorch_lightning.LightningModule` that automatically configures
        optimizers and learning rate scheduling based on a `Hydra <https://hydra.cc/>`_ configuration.

        The Hydra config should have an ``optimizer`` section, and optionally a ``schedule`` section
        if learning rate scheduling is desired.

        Sample Hydra Config

        .. code-block:: yaml

            optimizer:
              name: adam
              cls: torch.optim.Adam
              params:
                lr: 0.001

            schedule:
              interval: step
              monitor: val_loss
              frequency: 1
              cls: torch.optim.lr_scheduler.OneCycleLR
              params:
                max_lr: ${optimizer.params.lr}
                epochs: 10
                steps_per_epoch: 'none'
                pct_start: 0.03
                div_factor: 10
                final_div_factor: 10000.0
                anneal_strategy: cos
        """

        if not hasattr(self, "config"):
            raise AttributeError("'config' attribute is required for configure_optimizers")

        lr = self.config.optimizer["params"]["lr"]
        optim = instantiate(self.config.optimizer, self.parameters())

        # scheduler setup
        schedule = self.config.get("schedule")
        if schedule is not None:
            # interval/frequency keys required or schedule wont run
            for required_key in ["interval", "frequency"]:
                if required_key not in schedule.keys():
                    raise MisconfigurationException(f"{required_key} key is required for hydra lr scheduler")

            # monitor key is only required for schedules that watch loss
            if "monitor" not in schedule.keys():
                warnings.warn("'monitor' missing from lr schedule config")

            steps_per_epoch = len(self.train_dataloader())
            schedule_dict = {
                "interval": schedule.get("interval", "epoch"),
                "monitor": schedule.get("monitor", "val_loss"),
                "frequency": schedule.get("frequency", 1),
                "scheduler": instantiate(schedule, optim, max_lr=lr, steps_per_epoch=steps_per_epoch),
            }
            result = [optim], [schedule_dict]
        else:
            result = optim

        return result

    def prepare_data(self) -> None:
        r"""Override for :class:`pytorch_lightning.LightningModule` that automatically prepares
        any datasets based on a `Hydra <https://hydra.cc/>`_ configuration.

        The Hydra config should have an ``dataset`` section, and optionally a ``schedule`` section
        if learning rate scheduling is desired.

        Sample Hydra Config

        .. code-block:: yaml

            dataset:
              train:
                # passed to DataLoader
                num_workers: 1
                pin_memory: true
                drop_last: true
                shuffle: true

                # instantiates dataset
                cls: torchvision.datasets.FakeData
                params:
                  size: 10000
                  image_size: [1, 128, 128]
                  transform:
                    cls: torchvision.transforms.ToTensor

              # test/validation sets can be explicitly given as above,
              # or as a split from training set

              # as a random split from training set by number of examples
              # validate: 32

              # as a random split from training set by fraction
              # test: 0.1
        """
        dataset_cfg = self.config.dataset
        train_ds: Optional[Dataset] = (
            HydraMixin.instantiate(dataset_cfg["train"]) if "train" in dataset_cfg.keys() else None
        )

        # determine sizes validation/test sets if specified as a fraction of training set
        splits = {"train": len(train_ds) if train_ds is not None else None, "validate": None, "test": None}
        for split in splits.keys():
            if not (split in dataset_cfg.keys() and isinstance(dataset_cfg[split], (int, float))):
                continue
            if train_ds is None:
                raise MisconfigurationException("train dataset is required to perform splitting")

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

    def train_dataloader(self) -> Optional[DataLoader]:
        return self._dataloader(self.train_ds, "train")

    def val_dataloader(self) -> Optional[DataLoader]:
        return self._dataloader(self.val_ds, "validate")

    def test_dataloader(self) -> Optional[DataLoader]:
        return self._dataloader(self.test_ds, "test")

    def _dataloader(self, dataset: Optional[Dataset], split: str) -> Optional[DataLoader]:
        if dataset is not None:
            # try loading keys from dataset config
            assert split in self.config.dataset, f"split {split} missing from dataset config"
            if isinstance(self.config.dataset[split], (DictConfig, dict)):
                dataset_config = dict(self.config.dataset[split])

            # fallback to using training config, but force shuffle=False for test/val
            else:
                dataset_config = dict(self.config.dataset["train"])
                dataset_config["shuffle"] = False

            num_workers = dataset_config.get("num_workers", 1)
            pin_memory = dataset_config.get("pin_memory", False)
            drop_last = dataset_config.get("drop_last", False)
            shuffle = dataset_config.get("shuffle", False)

            return DataLoader(
                dataset,
                num_workers=num_workers,
                batch_size=self.hparams["batch_size"],
                shuffle=shuffle,
                pin_memory=pin_memory,
                drop_last=drop_last,
            )
        else:
            return None

    @staticmethod
    def instantiate(config: Union[DictConfig, dict], *args, **kwargs) -> Any:
        r"""
        Recursively instantiates classes in a Hydra configuration.

        Args:
            config (omegaconf.DictConfig or dict):
                The config to recursively instantiate from.

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
