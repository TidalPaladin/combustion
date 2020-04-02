#!/usr/bin/env python
# -*- coding: utf-8 -*-
import glob
import logging
import os
from argparse import Namespace
from functools import partial
from pathlib import Path
from typing import Tuple

import scipy.io as sio
import torch
import torch.utils.data as data
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split

from .dataset import MatlabDataset, MBBatch
from .preprocessing import power_of_two_crop


def load_data(args: Namespace, split: str) -> Dataset:
    if split not in ["train", "test"]:
        raise ValueError("Split must be train, test")
    logging.info("Loading %s data from %s", split, args.data_path)

    raw_ds = load_from_args(args)

    def to_loader(x):
        if args.distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(
                x, num_replicas=args.world_size, rank=args.rank, shuffle=True
            )
        else:
            sampler = None

        return DataLoader(
            x,
            batch_size=args.batch_size,
            pin_memory=True,
            collate_fn=MBBatch.collate_fn,
            shuffle=(sampler is None),
            sampler=sampler,
        )

    if split == "test":
        return to_loader(raw_ds)

    if args.validation_split is not None:
        len_val = int(len(raw_ds) * args.validation_split)
        len_train = len(raw_ds) - len_val
        train, val = random_split(raw_ds, [len_train, len_val])
    elif args.validation_size is not None:
        len_val = args.validation_size
        len_train = len(raw_ds) - len_val
        train, val = random_split(raw_ds, [len_train, len_val])
    elif args.validation_path is not None:
        train, val = raw_ds, load_from_args(args, "val")
    else:
        train, val = raw_ds, None

    if args.steps_per_epoch:
        logging.info("Selecting training subset based on steps_per_epoch=%d", args.steps_per_epoch)
        size = args.steps_per_epoch * args.batch_size
        train, _ = data.random_split(train, [size, len(train) - size])

    train = to_loader(train)
    val = to_loader(val) if val is not None else None
    return train, val




def load_from_args(args: Namespace, split="train") -> Dataset:
    if split == "train":
        target_path = args.data_path
    else:
        target_path = args.validation_path
    logging.info("Loading files from %s", target_path)
    files = list(Path(target_path).rglob("*.mat"))
    if not files:
        raise FileNotFoundError("no matching files in %s" % args.data_path)
    subsets = []
    for filename in files:
        subsets.append(MatlabDataset.from_args(args, filename))
    if len(subsets) > 1:
        return ConcatDataset(subsets)
    else:
        return subsets[0]
