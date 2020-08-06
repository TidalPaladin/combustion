#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
from copy import deepcopy
from itertools import islice
from typing import Any, Iterable, Optional, Union

import pytorch_lightning as pl
import torch
from hydra.errors import HydraException
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch.utils.data import DataLoader, Dataset, random_split


class HydraMixin:
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
    _has_inspected: bool = False
    train_ds: Optional[Dataset] = None
    val_ds: Optional[Dataset] = None
    test_ds: Optional[Dataset] = None

    def __new__(cls, *args, **kwargs):
        # because HydraMixin provides default overrides for test/val dataloader methods,
        # lightning will complain if a test/val step methods aren't provided
        #
        # to handle this, we replace test/val dataloader methods if test/val step methods
        # arent overridden
        if cls.validation_step == pl.LightningModule.validation_step:
            cls.val_dataloader = pl.LightningModule.val_dataloader
        else:
            cls.val_dataloader = HydraMixin.val_dataloader
        if cls.test_step == pl.LightningModule.test_step:
            cls.test_dataloader = pl.LightningModule.test_dataloader
        else:
            cls.test_dataloader = HydraMixin.test_dataloader

        x = super(HydraMixin, cls).__new__(cls)
        return x

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

    def prepare_data(self, force: bool = False) -> None:
        r"""Override for :class:`pytorch_lightning.LightningModule` that automatically prepares
        any datasets based on a `Hydra <https://hydra.cc/>`_ configuration.
        The Hydra config should have an ``dataset`` section, and optionally a ``schedule`` section
        if learning rate scheduling is desired.

        The following keys can be provided to compute statistics on the training set
            * ``stats_sample_size`` - number of training examples to compute statistics over,
              or ``"all"`` to use the entire training set.

            * ``stats_dim`` - dimension that will not be reduced when computing statistics.
              This is typically used when examples have more items than a simple ``(input, target)``
              tuple, such as when working with masks.
              Defaults to ``0``.

            * ``stats_index`` - tuple index of the data to compute statistics for.
              Defaults to ``0``.

        The following statistics will be computed and attached as attributes if ``stats_sample_size`` is set
            * ``channel_mean``
            * ``channel_variance``
            * ``channel_min``
            * ``channel_max``

        .. note::
            Training set statistics will be computed and attached when :func:`prepare_data` the first time.
            Subsequent calls will not alter the attached statistics.

        Args:

            force (bool):
                By default, training datasets will only be loaded once. When ``force=True``, datasets
                will always be reloaded.

        Sample Hydra Config

        .. code-block:: yaml

            dataset:
              stats_sample_size: 100    # compute training set statistics using 100 examples
              stats_dim: 0              # channel dimension to compute statistics for
              stats_index: 0            # tuple index to select from yielded example
              train:
                # passed to DataLoader
                num_workers: 1
                pin_memory: true
                drop_last: true
                shuffle: true

                # instantiates dataset
                target: torchvision.datasets.FakeData
                params:
                  size: 10000
                  image_size: [1, 128, 128]
                  transform:
                    target: torchvision.transforms.ToTensor

              # test/validation sets can be explicitly given as above,
              # or as a split from training set

              # as a random split from training set by number of examples
              # validate: 32

              # as a random split from training set by fraction
              # test: 0.1
        """
        if not force and self.train_ds is not None:
            return

        dataset_cfg = self.config.get("dataset")

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

        # collect sample statistics from the train ds and attach to self as a buffer
        if train_ds is not None and "stats_sample_size" in dataset_cfg.keys():
            num_examples = dataset_cfg["stats_sample_size"]
            if num_examples == "all":
                num_examples = len(train_ds)
            elif isinstance(num_examples, (float, int)):
                num_examples = int(num_examples)
            else:
                raise MisconfigurationException(f"Unexpected value for dataset.stats_sample_size: {num_examples}")

            dim = int(dataset_cfg["stats_dim"]) if "stats_dim" in dataset_cfg.keys() else 0
            index = int(dataset_cfg["stats_index"]) if "stats_index" in dataset_cfg.keys() else 0

            if num_examples < 0:
                raise MisconfigurationException(f"Expected dataset.stats_sample_size >= 0, found {num_examples}")
            elif num_examples > 0:
                self._inspect_dataset(train_ds, num_examples, dim, index)
            else:
                warnings.warn("dataset.stats_sample_size = 0, not collecting dataset staistics")

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
            batch_size = dataset_config.get("batch_size", self.hparams["batch_size"])
            collate_fn = dataset_config.get("collate_fn", None)

            return DataLoader(
                dataset,
                num_workers=num_workers,
                batch_size=batch_size,
                shuffle=shuffle,
                pin_memory=pin_memory,
                drop_last=drop_last,
                collate_fn=collate_fn,
            )
        else:
            return None

    def _inspect_dataset(self, dataset: Dataset, sample_size: int, dim: int = 0, index: int = 0) -> None:
        if self._has_inspected:
            return

        # get an example from dataset and ensure it is an iterable
        try:
            _ = dataset[0]
            if not isinstance(_, Iterable):
                _ = (_,)
        except RuntimeError:
            raise MisconfigurationException("Failed to get training example for statistics collection")

        # get the tensor requested by `index`
        try:
            _ = _[index]
        except IndexError:
            raise MisconfigurationException(
                f"Statistics requested for index {index}, but example had only {len(_)} elements"
            )

        # convert reverse (negative) indexing to forward indexing
        ndims = _.ndim
        original_dim = dim
        if dim < 0:
            dim = ndims + dim

        # get permutation such that the non-reduced dimension is first
        permuted_dims = [dim] + list(set(range(ndims)) - set([dim]))

        if dim >= ndims or dim < 0:
            raise MisconfigurationException(
                f"Statistics requested over dimension {original_dim}, but only {ndims} dims are present"
            )
        num_channels = _.shape[dim]

        # randomly draw sample_size examples from the given dataset and permute
        examples = torch.cat(
            [x[index].permute(*permuted_dims).view(num_channels, -1) for x in islice(iter(dataset), sample_size)], -1
        )

        # compute sample statistics and attach to self as a buffer
        var, mean = torch.var_mean(examples, dim=-1)
        maximum, minimum = examples.max(dim=-1).values, examples.min(dim=-1).values
        self.register_buffer("channel_mean", mean)
        self.register_buffer("channel_variance", var)
        self.register_buffer("channel_max", maximum)
        self.register_buffer("channel_min", minimum)
        self._has_inspected = True

    @staticmethod
    def instantiate(config: Union[DictConfig, dict], *args, **kwargs) -> Any:
        r"""
        Recursively instantiates classes in a Hydra configuration.

        Args:
            config (omegaconf.DictConfig or dict):
                The config to recursively instantiate from.

        """
        # NOTE this method is really ugly, but it's coming in Hydra 1.1 so no reason
        # to rewrite it
        if isinstance(config, dict):
            config = DictConfig(config)
        elif isinstance(config, list):
            config = ListConfig(config)

        # deepcopy so we can modify config
        config = deepcopy(config)
        if isinstance(config, DictConfig):
            OmegaConf.set_struct(config, False)

        if "params" in config.keys():
            if isinstance(config.get("params"), (dict, DictConfig)):
                params = dict(config.get("params"))
            elif isinstance(config.get("params"), (list, ListConfig)):
                params = list(config.get("params"))
        else:
            params = {}

        def is_subclass(d):
            if isinstance(d, (dict, DictConfig)):
                return "cls" in d.keys() or "target" in d.keys()
            elif isinstance(d, (list, ListConfig)):
                return any(["target" in x.keys() or "cls" in x.keys() for x in d if isinstance(x, (dict, DictConfig))])
            return False

        if isinstance(params, list):
            for i, subconfig in enumerate(params):
                subconfig = DictConfig(subconfig)
                if isinstance(subconfig, (dict, DictConfig)) and "params" not in subconfig.keys():
                    subconfig["params"] = {}
                subconfig._set_parent(config)
                params[i] = HydraMixin.instantiate(subconfig)
            del config["params"]
            return instantiate(config, *params, *args, **kwargs)

        else:
            subclasses = {key: subconfig for key, subconfig in params.items() if is_subclass(subconfig)}

            # instantiate recursively, remove those keys from config used in hydra instantiate call
            for key, subconfig in subclasses.items():
                subconfig = DictConfig(subconfig)
                # avoid issues when cls given without params
                if "params" not in subconfig:
                    subconfig["params"] = {}
                subconfig._set_parent(config)
                subclasses[key] = HydraMixin.instantiate(subconfig)
                del config.get("params")[key]

            subclasses.update(kwargs)

            # direct call to hydra instantiate
            try:
                return instantiate(config, *args, **subclasses)

            # hydra gives poor exception info
            # try a manual import of failed target and report the real error
            except HydraException as ex:
                import re

                msg = str(ex)
                target_name = re.search(r"'\S+'", msg).group().replace("'", "")
                try:
                    __import__(target_name)
                except RuntimeError:
                    raise

                # raise the original exception if import works
                raise
