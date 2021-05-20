#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch.nn import functional as F
from combustion.lightning.mixins import OptimizerMixin
import combustion
from dataclasses import dataclass
from hydra_configs.torch.optim import AdamConf
from hydra_configs.torch.optim.lr_scheduler import OneCycleLRConf
from hydra_configs.torch.utils.data.dataloader import DataLoaderConf
from typing import Any
from torchvision.datasets import FakeData
import combustion

from hydra.core.config_store import ConfigStore
from .conf import *

log = logging.getLogger(__name__)


#combustion.initialize(config_path="./conf", config_name="config")


@hydra.main(config_path="./conf", config_name="config")
def main(cfg):
    print(cfg.to_yaml())
    return combustion.main(cfg)


if __name__ == "__main__":
    main()
    combustion.check_exceptions()
