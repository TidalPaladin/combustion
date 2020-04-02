#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABC, abstractclassmethod
from collections import OrderedDict
from itertools import islice
from typing import Callable, Generator, Iterable, List, Optional, Tuple, Iterator

import torch
from torch import Tensor

from combustion.util import Dim

class Batch(ABC):
    r"""Abstract base class representing a batch of examples. Items are
    passed to the batch constructor as keyword args. 

    Keyword args are processed as follows:
        * Tensor:  tensor is attached to the batch without modification
        * List of Tensors: Tensors in the list are stacked along a new 
        dimension with index 0
        * Tuple of Tensors: Same behavior as with list of tensors

    The following features are supported:
        * Tuple expansion based on the order of keyword args given at initialization.
        * Application of Tensor operations to all tensors in batch via `apply`

    The following features are experimental:
        * Application of Tensor operations to all tensors in batch via `getattr`
    """

    def __init__(self, **kwargs):
        if not kwargs:
            raise ValueError("keyword args must be given to batch constructor")

        self._tensors = OrderedDict()
        for k, v in kwargs.items():
            # if kwarg given list of examples, stack along new batch dim 0
            if isinstance(v, (list, tuple)):
                v = [torch.as_tensor(elem) for elem in v]
                v = torch.stack(v, Dim.BATCH)
            elif not isinstance(v, Tensor):
                raise TypeError(f"expected Tensor for {k}, found {type(v)}")

            self._tensors[k] = v

            batch_size = len(self)
            if len(v) != batch_size:
                raise ValueError(f'expected tensors of equal batch size: {len(v)} vs {batch_size}')

    def __iter__(self) -> Iterator[Tensor]:
        """Iterates over batch tensors using order of keyword args when created"""
        return iter(self._tensors.values())

    def __getitem__(self, pos: int) -> Tuple[Tensor]:
        """Slices the `pos`th example from the batch"""
        return tuple([x[pos] for x in self._tensors.values()])

    def __len__(self):
        """Gets the number of examples in the batch"""
        return len(next(iter(self._tensors.values())))

    def __repr__(self):
        name = self.__class__.__name__
        s = f"{name}("
        it = iter(self._tensors.items())
        k, v = next(it, None)
        if k:
            s += f"{k}=({tuple(v.shape)})"
        for k, v in it:
            s += f", {k}=({tuple(v.shape)})"
        s += ")"
        return s

    def __getattr__(self, attr):
        """
        Checks if the requested attribute exists on torch.Tensor and the Tensor 
        attribute is a callable. If so, the Tensor callable is invoked on all batch
        tensors. This is an experimental shortcut that allows for operations like
        `batch.cuda()`
        """
        if attr in self._tensors:
            return self._tensors[attr]
        elif hasattr(Tensor, attr) and hasattr(getattr(Tensor, attr), "__call__"):
            def func(*args, **kwargs):
                new_tensors = OrderedDict()
                for k, v in self._tensors.items():
                    new_v = getattr(v, attr)(*args, **kwargs)
                    new_tensors[k] = new_v if new_v is not None else v
                self._tensors.update(new_tensors)
                return self
            return func
        else:
            raise AttributeError(f"{attr}")

    def apply(self, func: Callable[[Tensor], Optional[Tensor]]) -> None:
        """Applies a function to all tensors in the batch."""
        new_tensors = OrderedDict()
        for k, v in self._tensors.items():
            new_v = func(v)
            new_tensors[k] = new_v if new_v is not None else v
        self._tensors.update(new_tensors)

    @classmethod
    def collate_fn(cls, examples: Iterable[Tensor]):
        r"""Collate function used to collect an iterable of examples into 
        a batch. 

        See https://pytorch.org/docs/stable/data.html#working-with-collate-fn
        for more details.
        """
        raise NotImplementedError("must implement collate_fn")
