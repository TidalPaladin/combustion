#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import warnings
from copy import deepcopy
from itertools import islice
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
from hydra.errors import HydraException
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, random_split


_INSTANTIATE_KEYS = ["cls", "target", "_target_"]


def _is_instantiate_key(k: str) -> bool:
    return k in _INSTANTIATE_KEYS


def _has_instantiate_key(i: Iterable) -> bool:
    return any([_is_instantiate_key(k) for k in i])


class HydraModule(pl.LightningModule):
    r"""
    Module for creating :class:`pytorch_lightning.LightningModule`
    using `Hydra <https://hydra.cc/>`_.

    The following :class:`pytorch_lightning.LightningModule` abstract methods are implemented:

        * :attr:`configure_optimizers`
        * :attr:`prepare_data`
        * :attr:`train_dataloader`
        * :attr:`val_dataloader`
        * :attr:`test_dataloader`

    Accessors are provided for Pytorch Lightning ``hparams`` field:

        * :attr:`hparams` - Returns ``config.model.params`` from the Hydra config.
        * :attr:`instantiated_hparams` - Returns the ``**kwargs`` used when instantiating the model.
          When Hydra recursive instantiation is used, result will contain instantiated values.
        * :attr:`instantiated_hparams.setter` - Should be assigned the kwarg dict used when calling
          ``model.__init__()``.
        * :attr:`hparams.setter` - Equivalent to ``instantiated_hparams.setter``.

    Usage Pattern:
        >>> class MyModel(HydraModule):
        >>>
        >>>     def __init__(self, config, **hparams):
        >>>         super().__init__(config, **hparams)
        >>>         # your model code here
        >>>         ...

    Alternate Usage Pattern (preserves operability without Hydra / Pytorch Lightning):
        >>> class MyModel(nn.Module):
        >>>     def __init__(self, arg1, arg2, ...):
        >>>         # your model code here
        >>>         ...
        >>>
        >>> class MyHydraModel(MyModel, HydraModule):
        >>>     def __init__(self, config, **hparams):
        >>>         # call MyModel/HydraModule init here
        >>>         ...
    """
    _has_inspected: bool = False
    _has_datasets: bool = False
    train_ds: Optional[Dataset] = None
    val_ds: Optional[Dataset] = None
    test_ds: Optional[Dataset] = None
    _config: Optional[DictConfig] = None
    _hparams: Optional[DictConfig] = None

    def __init__(self, config: DictConfig, **hparams):
        super().__init__()
        self.config = config
        self.hparams = hparams

    @property
    def hparams(self) -> Optional[DictConfig]:
        if "model" in self._config.keys():
            return self._config.model.get("params", None)
        return None

    @hparams.setter
    def hparams(self, val: Union[Dict[str, Any], DictConfig]) -> None:
        if isinstance(val, DictConfig):
            val = dict(val)
        self._hparams = val

    @property
    def instantiated_hparams(self) -> Optional[Dict[str, Any]]:
        return self._hparams

    @instantiated_hparams.setter
    def instantiated_hparams(self, val: DictConfig) -> None:
        self._hparams = val

    @property
    def config(self) -> Optional[DictConfig]:
        return self._config

    @config.setter
    def config(self, val: DictConfig):
        self._config = val

    def __new__(cls, *args, **kwargs):
        # because HydraModule provides default overrides for test/val dataloader methods,
        # lightning will complain if a test/val step methods aren't provided
        #
        # to handle this, we replace test/val dataloader methods if test/val step methods
        # arent overridden
        if cls.validation_step == pl.LightningModule.validation_step:
            cls.val_dataloader = pl.LightningModule.val_dataloader
        else:
            cls.val_dataloader = HydraModule.val_dataloader
        if cls.test_step == pl.LightningModule.test_step:
            cls.test_dataloader = pl.LightningModule.test_dataloader
        else:
            cls.test_dataloader = HydraModule.test_dataloader

        x = super(HydraModule, cls).__new__(cls)
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

            test_only = self.config.trainer.get("test_only", False)
            dl = self.train_dataloader()
            if dl is not None:
                # get gpus / num_nodes / accum_grad_batches to compute optimizer steps per epoch
                accum_grad_batches = int(self.config.trainer["params"].get("accumulate_grad_batches", 1))
                assert accum_grad_batches >= 1
                gpus = self.config.trainer["params"].get("gpus", 1)
                if isinstance(gpus, Iterable):
                    gpus = len(gpus)
                gpus = max(gpus, 1)
                num_nodes = self.config.trainer["params"].get("num_nodes", 1)
                steps_per_epoch = math.ceil(len(dl) / (accum_grad_batches * gpus * num_nodes))

                schedule_dict = {
                    "interval": schedule.get("interval", "epoch"),
                    "monitor": schedule.get("monitor", "val_loss"),
                    "frequency": schedule.get("frequency", 1),
                    "scheduler": instantiate(schedule, optim, max_lr=lr, steps_per_epoch=steps_per_epoch),
                }
                result = [optim], [schedule_dict]

            else:
                if not test_only:
                    raise RuntimeError("Could not create LR schedule because train_dataloader() returned None")
                result = optim
        else:
            result = optim
        return result

    def get_datasets(self, force: bool = False) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        r"""Automatically prepares train/validation/test datasets based on a `Hydra <https://hydra.cc/>`_ configuration.
        The Hydra config should have an ``dataset`` section, and optionally a ``schedule`` section
        if learning rate scheduling is desired.

        .. warning::
            Statistics collection is deprecated and will be removed in a later release. Use
            :class:`torch.nn.BatchNorm2d` with ``affine=False`` to collect running means/variances.

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
                By default, datasets will only be loaded once and cached for subsequent calls.
                When ``force=True``, datasets will always be reloaded.

        Returns:
            ``(training_dataset, validation_dataset, test_dataset)``

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
        if self._has_datasets and not force:
            return self.train_ds, self.val_ds, self.test_ds

        dataset_cfg = self.config.get("dataset")

        # when in test_only mode, try to avoid setting up training/validation sets
        # training set is only needed if test set is a split from training
        test_only = "test_only" in self.config.trainer and self.config.trainer["test_only"]
        if test_only and not isinstance(dataset_cfg["test"], (int, float)):
            test_ds = HydraModule.instantiate(dataset_cfg["test"]) if "test" in dataset_cfg.keys() else None
            self.train_ds = None
            self.val_ds = None
            self.test_ds = test_ds
            self._has_datasets = True
            return self.train_ds, self.val_ds, self.test_ds

        train_ds: Optional[Dataset] = (
            HydraModule.instantiate(dataset_cfg["train"]) if "train" in dataset_cfg.keys() else None
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
            test_ds = HydraModule.instantiate(dataset_cfg["test"]) if "test" in dataset_cfg.keys() else None
        elif splits["test"] is not None:
            lengths = (splits["train"], splits["test"])
            train_ds, test_ds = random_split(train_ds, lengths)
            val_ds = HydraModule.instantiate(dataset_cfg["validate"]) if "validate" in dataset_cfg.keys() else None
        else:
            val_ds = HydraModule.instantiate(dataset_cfg["validate"]) if "validate" in dataset_cfg.keys() else None
            test_ds = HydraModule.instantiate(dataset_cfg["test"]) if "test" in dataset_cfg.keys() else None

        # collect sample statistics from the train ds and attach to self as a buffer
        if train_ds is not None and "stats_sample_size" in dataset_cfg.keys():
            warnings.warn(
                "stats_sample_size is deprecated, use torch.nn.BatchNorm(affine=False) instead", DeprecationWarning
            )
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

        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds
        self._has_datasets = True
        return self.train_ds, self.val_ds, self.test_ds

    def train_dataloader(self) -> Optional[DataLoader]:
        train_ds, _, _ = self.get_datasets()
        return self._dataloader(train_ds, "train")

    def val_dataloader(self) -> Optional[DataLoader]:
        _, val_ds, _ = self.get_datasets()
        return self._dataloader(val_ds, "validate")

    def test_dataloader(self) -> Optional[DataLoader]:
        _, _, test_ds = self.get_datasets()
        return self._dataloader(test_ds, "test")

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

        self.register_buffer("channel_mean", mean.to(self.device))
        self.register_buffer("channel_variance", var.to(self.device))
        self.register_buffer("channel_max", maximum.to(self.device))
        self.register_buffer("channel_min", minimum.to(self.device))
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
                return _has_instantiate_key(d.keys())
            elif isinstance(d, (list, ListConfig)):
                return any([_has_instantiate_key(x.keys()) for x in d if isinstance(x, (dict, DictConfig))])
            return False

        if isinstance(params, list):
            for i, subconfig in enumerate(params):
                subconfig = DictConfig(subconfig)
                if isinstance(subconfig, (dict, DictConfig)) and "params" not in subconfig.keys():
                    subconfig["params"] = {}
                subconfig._set_parent(config)
                params[i] = HydraModule.instantiate(subconfig)
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
                subclasses[key] = HydraModule.instantiate(subconfig)
                del config.get("params")[key]

            subclasses.update(kwargs)

            try:
                # direct call to hydra instantiate
                return instantiate(config, *args, **subclasses)

            # hydra gives poor exception info
            # try a manual import of failed target and report the real error
            except HydraException as ex:
                import re

                msg = str(ex)
                target_name = re.search(r"'\S+'", msg).group().replace("'", "")
                try:
                    __import__(target_name)
                except SyntaxError:
                    raise
                except RuntimeError:
                    pass

                # raise the original exception if import works
                raise
