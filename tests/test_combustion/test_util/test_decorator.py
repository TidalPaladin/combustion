#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytorch_lightning as pl
import torch
import torch.nn as nn
from hydra.utils import instantiate
from torch import Tensor

from combustion.util import dataclass_init, hydra_dataclass


@hydra_dataclass(target="MyModel.from_args")
class MyModelConf:
    in_features: int
    out_features: int
    kernel: int = 3
    num_repeats: int = 3


@dataclass_init(MyModelConf)
class MyModel(pl.LightningModule):
    def __init__(self, conf: MyModelConf):
        super().__init__()
        self.conf = conf
        self.repeats = nn.Sequential()
        self.repeats.add_module("conv_1", nn.Conv2d(conf.in_features, conf.out_features, conf.kernel))
        for i in range(conf.num_repeats):
            self.repeats.add_module("conv_{i+1}", nn.Conv2d(conf.out_features, conf.out_features, conf.kernel))
        self.save_hyperparameters(conf.to_omegaconf())

    def forward(self, inputs: Tensor) -> Tensor:
        _ = self.conv(inputs)
        return self.repeats(_)


class TestDataclasses:
    def test_dataclass_init(self):
        conf = MyModelConf(in_features=10, out_features=10)

        model = MyModel.from_args(in_features=conf.in_features, out_features=conf.out_features)
        assert isinstance(model, MyModel)
        assert model.conf == conf

        model = MyModel(conf)
        assert isinstance(model, MyModel)
        assert model.conf == conf

        model = instantiate(conf)
        assert isinstance(model, MyModel)
        assert model.conf == conf

    def test_dataclass_init_save_hparams(self):
        conf = MyModelConf(in_features=10, out_features=10)
        model = MyModel(conf)
        assert model.hparams == conf

    def test_config_store(self):
        # TODO all this does is run with name/group assignment
        @hydra_dataclass(target="foo", name="base_mymodelconf", group="model")
        class MyModelConf:
            in_features: int
            out_features: int
            kernel: int = 3
            num_repeats: int = 3

    def test_make_dataclass(self):
        @hydra_dataclass(spec=pl.Trainer)
        class TrainerConf:
            ...

        conf = TrainerConf()
        trainer = instantiate(conf)
        assert isinstance(trainer, pl.Trainer)

    def test_make_dataclass_no_default(self):
        @hydra_dataclass(spec=torch.optim.Adam)
        class AdamConf:
            ...

        conf = AdamConf()
        model = nn.Linear(10, 10)
        adam = instantiate(conf, params=model.parameters())
        assert isinstance(adam, torch.optim.Adam)

        conf.params = model.parameters()
        adam = instantiate(conf)
        assert isinstance(adam, torch.optim.Adam)

    def test_make_dataclass_recursive(self):
        class ProtoClass:
            def __init__(self, x: int = 1, y: nn.Module = nn.Linear(10, 10)):
                self.x = x
                self.y = y

        @hydra_dataclass(spec=ProtoClass, recursive=True)
        class ProtoClassConf:
            ...

        conf = ProtoClassConf()
