#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
from argparse import Namespace

import torch.nn as nn


def get_model(args: Namespace) -> nn.Module:
    logging.info("Loading model %s", args.model)

    if args.model == "model1":
        model = nn.Conv2d(10, 10, 3)
    elif args.model == "model2":
        model = nn.Linear(10, 1)
    else:
        raise ValueError(f"unknown model arg: {args.model}")

    logging.info("Loaded model instance %s", model.__class__.__name__)
    return model
