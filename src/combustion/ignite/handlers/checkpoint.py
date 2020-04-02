#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import os
from argparse import Namespace
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Union

import ignite.handlers as handlers
import torch
from ignite.engine import Engine, Events
from torch import Tensor
from torch.nn import Module


class CheckpointLoader:
    r"""
    Handles resuming from a model checkpoint, either from a filepath or by most recent file.

    Args:
        model (nn.Module): Model to load
        filepath (str, optional): Path to checkpoint file or directory of checkpoint files
        resume (bool, optional): If true, resume from the most recent file in `filepath`
        strict (bool, optional): If false, ignore keyword args that are not present in loaded checkpoint
    """

    def __init__(self, filepath, resume=False, strict=True, **to_load):
        # type: (Module, str, bool)
        if not isinstance(resume, bool):
            raise TypeError(f"resume must be bool, found {type(resume)}")
        if not isinstance(strict, bool):
            raise TypeError(f"strict must be bool, found {type(strict)}")
        if not to_load:
            raise ValueError("must give one or more load targets as keyword args")
        for k, v in to_load.items():
            if not hasattr(v, "load_state_dict"):
                raise ValueError(f"key {k}={type(v)} has no load_state_dict method")

        self.to_load = to_load
        self.filepath = filepath
        self.resume = resume
        self.strict = strict

    def __repr__(self):
        # type: () -> str
        name = self.__class__.__name__
        s = f"{name}({self.filepath}, to_load={self.to_load.keys()}"
        if self.resume:
            s += f", resume=True"
        if not self.strict:
            s += f", strict=False"
        return s + ")"

    @staticmethod
    def newest_file(path, glob="*.pth", recursive=True):
        # type: (str, str, bool) -> Optional[str]
        r"""
        Gets the newest file in a directory by glob

        Args:
            path (str): Base path to search
            glob (str, optional): Glob pattern to match against files
            recursive (bool, optional): If true, search for glob matches recursively

        Returns:
            The most recent file in `path` matching `glob`, or None if no match 
            was found. 
        """
        if recursive:
            files = [p for p in Path(path).rglob(glob)]
        else:
            files = [p for p in Path(path).glob(glob)]

        files = list(filter(Path.is_file, files))
        file_list = sorted(files, key=lambda x: x.stat().st_mtime, reverse=True)
        return file_list[0] if file_list else None

    def __call__(self, engine):
        # type: Engine
        # get checkpoint file target
        if self.resume:
            target = self.newest_file(self.filepath)
            if not target:
                logging.error("No checkpoint file in %s", self.filepath)
                return
        else:
            target = self.filepath
            if not os.path.isfile(target):
                raise FileNotFoundError(f"checkpoint file {target} not found")

        # restore all objects given to constructor kwargs
        state_dicts = torch.load(target)
        for name, obj in self.to_load.items():
            if name in state_dicts:
                obj.load_state_dict(state_dicts[name])
            elif self.strict:
                raise KeyError(f"key {name} not in checkpoint state dict")

        logging.info("Restored from checkpoint %s", target)

    @classmethod
    def from_args(cls, args, **to_load):
        # type: (Namespace) -> Optional[CheckpointLoader]
        if args.resume:
            path = Path(args.model_path).parent
        elif args.load_model:
            path = args.load_model
        else:
            return None
        return cls(path, resume=args.resume, **to_load)


class ModelCheckpoint(handlers.ModelCheckpoint):
    @classmethod
    def from_args(cls, args):
        # type: (Namespace) -> Optional[CheckpointLoader]
        return cls(
            dirname=args.model_path,
            filename_prefix=args.checkpoint_prefix,
            n_saved=args.n_saved,
            require_empty=(not args.overwrite),
        )

    def attach(self, engine, args, **to_save):
        # type: (Engine)
        if args.multiprocessing_distributed and args.rank % args.gpus_per_node != 0:
            return

        if args.checkpoint_steps:
            event = Events.ITERATION_COMPLETED(every=args.checkpoint_steps)
        elif args.checkpoint_epochs:
            event = Events.EPOCH_COMPLETED(every=args.checkpoint_epochs)
        else:
            return
        engine.add_event_handler(event, self, to_save)
