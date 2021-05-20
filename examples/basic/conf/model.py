#!/usr/bin/env python
# -*- coding: utf-8 -*-
from ..model import FakeModelConf, FakeModel
from combustion.util import hydra_dataclass

@hydra_dataclass(name="fakemodel2", group="model")
class OtherModelConf(FakeModelConf):
    kernel: int = 5
