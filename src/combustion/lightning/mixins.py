#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import warnings
from copy import deepcopy
from inspect import signature
from typing import Any, Generator, Iterable, Optional, Tuple, Union

import pytorch_lightning as pl
import torch.nn as nn
from hydra.errors import InstantiationException
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf
from pytorch_lightning.core.saving import ModelIO
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset, random_split


_INSTANTIATE_KEYS = ["cls", "target", "_target_"]


def _is_instantiate_key(k: str) -> bool:
    return k in _INSTANTIATE_KEYS


def _has_instantiate_key(i: Iterable) -> bool:
    return any([_is_instantiate_key(k) for k in i])


class HydraMixin(ModelIO):
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
        >>> class MyModel(HydraMixin):
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
        >>> class MyHydraModel(MyModel, HydraMixin):
        >>>     def __init__(self, config, **hparams):
        >>>         # call MyModel/HydraMixin init here
        >>>         ...
    """
    _has_datasets: bool = False
    train_ds: Optional[Dataset] = None
    val_ds: Optional[Dataset] = None
    test_ds: Optional[Dataset] = None

    @staticmethod
    def create_model(hparams):
        r"""Creates a model using Hydra instantiation. The returned
        object will be a subclass of the instantiated model, with overrides
        to facilitate hydra-compatible checkpoint loading.

        .. note::
            You must call ``self.save_hyperparameters()`` in your LightningModule's init method.

        Args:
            hparams (DictConfig):
                Hydra config to use when creating the model. Should at least provide
                a ``hparams.model`` key

        Returns:
            Subclass of the instantiated model with ``__init__`` overriden to support Hydra
            based checkpoint saving / loading.
        """
        # instantiate the actual model
        model = HydraMixin.instantiate(hparams.model)

        # create a wrapper class so that loading from checkpoint
        # can accept a hydra config rather than model level hparams
        class ModelWrapper(model.__class__):
            def __init__(self, **hparams):
                # warn user if model is not a HydraMixin
                if not isinstance(model, HydraMixin):
                    warnings.warn(
                        f"Model type {type(model)} is not a HydraMixin instance. "
                        "Default methods provided by combustion.lightning.HydraMixin will not be available"
                    )

                # check that save_hyperparameters was called
                has_params = bool(signature(model).parameters)
                if not hasattr(model, "hparams") or (has_params and not model.hparams):
                    import pdb

                    pdb.set_trace()
                    raise InstantiationException(
                        "Please call self.save_hyperparameters() in your model's __init__ method"
                    )

                # construct the wrapped model by reading instantiated hyperparams
                super().__init__(**model.hparams)
                self.__class__.__name__ = f"ModelWrapper<{model.__class__.__name__}>"

                # replace hparams property with hydra dictconfig
                self.hparams.clear()
                self.save_hyperparameters()

        return ModelWrapper(**hparams)

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
              _target_: torch.optim.Adam
              params:
                lr: 0.001

            schedule:
              interval: step
              monitor: val_loss
              frequency: 1
              _target_: torch.optim.lr_scheduler.OneCycleLR
              params:
                max_lr: ${optimizer.params.lr}
                epochs: 10
                steps_per_epoch: 'none'
                pct_start: 0.03
                div_factor: 10
                final_div_factor: 10000.0
                anneal_strategy: cos
        """
        scheduler = self.hparams.get("schedule", None)
        optim = self.hparams.get("optimizer")
        # multiple optimizers
        if isinstance(optim, ListConfig):
            num_optims = len(optim)
            optim_params = [self.get_optimizer_parameters(idx) for idx in range(num_optims)]
            optim_configs = optim

            if not scheduler:
                scheduler_configs = [None] * num_optims
            elif isinstance(scheduler, DictConfig):
                scheduler_configs = [scheduler] * num_optims
            elif isinstance(scheduler, ListConfig):
                scheduler_configs = scheduler
            else:
                raise TypeError(f"Unable to process scheduler config: \n{scheduler}")

            optims = []
            schedulers = []
            zipped = zip(optim_configs, optim_params, scheduler_configs)
            for optim_config, optim_param, schedule_config in zipped:
                result = self._configure_optimizers(optim_config, optim_param, schedule_config)
                if isinstance(result, Optimizer):
                    optims.append(result)
                elif isinstance(result, tuple) and len(result) == 2:
                    optims.append(result[0][0])
                    schedulers.append(result[1][0])
                else:
                    raise RuntimeError(f"Unable to process result: {result}")

            if schedulers:
                return optims, schedulers
            else:
                return optims

        # single optimizer
        else:
            return self._configure_optimizers(optim, self.parameters(), scheduler)

    def get_optimizer_parameters(self, optim_idx: int) -> Generator[nn.Parameter, None, None]:
        r"""Abstract method that must be implemented when using multiple optimizers. With multiple
        optimizers, the instantiation process is roughly:

            >>> optimizers = []
            >>> for optim_idx, optim_config in enumerate(optimizer_configs):
            >>>     params = self.get_optimizer_parameters(optim_idx)
            >>>     optim = self.instantiate(optim_config, params)
            >>>     optimizers.append(optim)

        Args:
            optim_idx (int):
                Index of the optimizer to retrieve associated module parameters for. Indices
                follow the order in which optimizers were specified in the optimizer config.

        Returns:
            Parameters to be optimized by the ``optim_idx``'th optimizer.'
        """
        if optim_idx != 0:
            raise NotImplementedError(
                "HydraMixin modules using multiple optimizers must implement `get_optimizer_parameters`."
            )
        return self.parameters()

    def _configure_optimizers(
        self,
        optim_config: DictConfig,
        params: Generator[nn.Parameter, None, None],
        schedule_config: Optional[DictConfig],
    ):
        lr = optim_config["params"]["lr"]
        optim = instantiate(optim_config, params)

        # scheduler setup
        if schedule_config is not None:
            # interval/frequency keys required or schedule wont run
            for required_key in ["interval", "frequency"]:
                if required_key not in schedule_config.keys():
                    raise MisconfigurationException(f"{required_key} key is required for hydra lr scheduler")

            # monitor key is only required for schedules that watch loss
            if "monitor" not in schedule_config.keys():
                warnings.warn("'monitor' missing from lr schedule config")

            test_only = self.hparams.trainer.get("test_only", False)
            dl = self.train_dataloader()
            if dl is not None:
                # get gpus / num_nodes / accum_grad_batches to compute optimizer steps per epoch
                accum_grad_batches = int(self.hparams.trainer["params"].get("accumulate_grad_batches", 1))
                assert accum_grad_batches >= 1
                gpus = self.hparams.trainer["params"].get("gpus", 1)
                if isinstance(gpus, Iterable):
                    gpus = len(gpus)
                gpus = max(gpus, 1)
                num_nodes = self.hparams.trainer["params"].get("num_nodes", 1)
                steps_per_epoch = math.ceil(len(dl) / (accum_grad_batches * gpus * num_nodes))

                schedule_dict = {
                    "interval": schedule_config.get("interval", "epoch"),
                    "monitor": schedule_config.get("monitor", "val_loss"),
                    "frequency": schedule_config.get("frequency", 1),
                    "scheduler": instantiate(schedule_config, optim, max_lr=lr, steps_per_epoch=steps_per_epoch),
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

        dataset_cfg = self.hparams.get("dataset")

        # when in test_only mode, try to avoid setting up training/validation sets
        # training set is only needed if test set is a split from training
        test_only = "test_only" in self.hparams.trainer and self.hparams.trainer["test_only"]
        if test_only and not isinstance(dataset_cfg["test"], (int, float)):
            test_ds = HydraMixin.instantiate(dataset_cfg["test"]) if "test" in dataset_cfg.keys() else None
            self.train_ds = None
            self.val_ds = None
            self.test_ds = test_ds
            self._has_datasets = True
            return self.train_ds, self.val_ds, self.test_ds

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
            assert split in self.hparams.dataset, f"split {split} missing from dataset config"
            if isinstance(self.hparams.dataset[split], (DictConfig, dict)):
                dataset_config = dict(self.hparams.dataset[split])

            # fallback to using training config, but force shuffle=False for test/val
            else:
                dataset_config = dict(self.hparams.dataset["train"])
                dataset_config["shuffle"] = False

            num_workers = dataset_config.get("num_workers", None) or self.hparams.dataset.get("batch_size", None) or 1
            pin_memory = dataset_config.get("pin_memory", False)
            drop_last = dataset_config.get("drop_last", False)
            shuffle = dataset_config.get("shuffle", False)
            collate_fn = dataset_config.get("collate_fn", None)
            batch_size = dataset_config.get("batch_size", self.hparams.dataset["batch_size"])

            if collate_fn is not None:
                collate_fn = self.instantiate(collate_fn)

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
                params[i] = HydraMixin.instantiate(subconfig)
            del config["params"]
            return instantiate(config, *params, *args, **kwargs)

        else:
            subclasses = {key: subconfig for key, subconfig in params.items() if is_subclass(subconfig)}

            # instantiate recursively, remove those keys from config used in hydra instantiate call
            for key, subconfig in subclasses.items():

                if isinstance(subconfig, (dict, DictConfig)):
                    subconfig = DictConfig(subconfig)
                    # avoid issues when cls given without params
                    if "params" not in subconfig:
                        subconfig["params"] = {}
                    subconfig._set_parent(config)
                    subclasses[key] = HydraMixin.instantiate(subconfig)
                else:
                    subclasses[key] = [HydraMixin.instantiate(x) for x in subconfig]

                del config.get("params")[key]

            subclasses.update(kwargs)
            return instantiate(config, *args, **subclasses)
