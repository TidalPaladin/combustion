#!/usr/bin/env python
# -*- coding: utf-8 -*-


from dataclasses import dataclass
from typing import Any, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from pytorch_lightning import Trainer

from combustion.util import hydra_dataclass
from abc import ABC



@hydra_dataclass(spec=Trainer, name="base_trainer", group="trainer")
class TrainerConf:
    ...


@dataclass
class CombustionConf:
    data: Any = MISSING
    trainer: TrainerConf = MISSING
    model: Any = MISSING
    seed: int = 42
    fit: bool = False
    test: bool = False
    predict: bool = False
    catch_exceptions: bool = False
    load_checkpoint: Optional[str] = None


cs = ConfigStore.instance()
cs.store(name="combustion", node=CombustionConf)
