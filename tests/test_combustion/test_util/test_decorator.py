#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from hydra.utils import instantiate
from omegaconf import MISSING
from torch import Tensor
from typing import Any
from dataclasses import dataclass
from hydra.core.config_store import ConfigStore

from combustion.util import dataclass_init, hydra_dataclass, Instantiable


class PrimitiveConf:
    x: int = 1
    y: int = 3


class PrimitiveModel:
    def __init__(self, x: int, y: int):
        self.conf = conf
        pass


class ComplexConf:
    x: int = 1
    y: int = 3
    obj: Any = MISSING


class ComplexModel(PrimitiveModel):

    def __init__(self, x: int, y: int, obj: Any):
        self.conf = conf

@dataclass
class SubConf:
    z: int = 4


class NestedConf:
    x: int = 1
    y: int = 3
    sub: SubConf = SubConf(z=3)


class TestHydraDataclass:

    def test_primitive(self):
        Conf = hydra_dataclass(target="PrimitiveModel")(PrimitiveConf)
        conf = Conf()
        assert isinstance(conf, PrimitiveConf)
        assert conf._target_ == "test_decorator.PrimitiveModel"
        assert conf.x == 1
        assert conf.y == 3

    def test_complex(self):
        Conf = hydra_dataclass(target="ComplexModel")(ComplexConf)
        conf = Conf()
        assert isinstance(conf, ComplexConf)
        assert conf._target_ == "test_decorator.ComplexModel"
        assert conf.x == 1
        assert conf.y == 3
        assert conf.obj == MISSING

    def test_nested(self):
        Conf = hydra_dataclass(target="ComplexModel")(NestedConf)
        conf = Conf()
        assert isinstance(conf, NestedConf)
        assert conf._target_ == "test_decorator.ComplexModel"
        assert conf.x == 1
        assert conf.y == 3
        assert conf.sub.z == 3

    def test_config_store(self):
        group = "foo"
        name = "bar"
        cs = ConfigStore.instance()
        assert group not in cs.repo.keys()
        Conf = hydra_dataclass(target="ComplexModel", name=name, group=group)(PrimitiveConf)
        assert group in cs.repo.keys()
        key = f"{name}.yaml" 
        assert key in cs.repo[group].keys()
        node = cs.repo[group][key]
        _node = node.node
        assert _node._target_ == "test_decorator.PrimitiveModel"
