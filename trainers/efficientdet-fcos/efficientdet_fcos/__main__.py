#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os

import hydra
import pytorch_lightning as pl
import torch

import combustion
import medcog_preprocessing
try:
    import medcog_classifier
except ModuleNotFoundError:
    import pdb; pdb.set_trace()


config_path = "../conf"
config_name = "config"
log = logging.getLogger(__name__)

if "WORLD_SIZE" not in os.environ:
    combustion.initialize(config_path=config_path, config_name=config_name)
    print("Called initialize")


@hydra.main(config_path=config_path, config_name=config_name)
def main(cfg):
    combustion.main(cfg)


if __name__ == "__main__":
    main()
