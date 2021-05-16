#!/usr/bin/env python
# -*- coding: utf-8 -*-


from dataclasses import dataclass
from typing import Any, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


# @dataclass
# class Trainer(Dict[str, Any]):
#    _target_: str = "pytorch_lightning.Trainer"


@dataclass
class CombustionConf:
    data: Any = MISSING
    trainer: Any = MISSING
    model: Any = MISSING
    seed: int = 42
    fit: bool = False
    test: bool = False
    predict: bool = False
    catch_exceptions: bool = False
    load_checkpoint: Optional[str] = None


cs = ConfigStore.instance()
cs.store(name="combustion", node=CombustionConf)
