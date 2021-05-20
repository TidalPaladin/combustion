#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pytorch_lightning import Trainer
from combustion.util import hydra_dataclass

@hydra_dataclass(spec=Trainer, name="trainer2", group="trainer")
class TrainerConf:
    ...
