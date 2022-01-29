#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import abstractclassmethod, abstractmethod, abstractproperty
from dataclasses import is_dataclass, replace
from itertools import chain
from typing import Any, Callable, ClassVar, Dict, Iterable, Iterator, List, Optional, Tuple, TypeVar, Union, cast

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning.utilities.apply_func import apply_to_collection, move_data_to_device
from torch import Tensor


T = TypeVar("T", bound="TensorDataclass")


@torch.jit.script
def pad(t: Tensor, shape: List[int], mode: str = "constant", value: float = 0) -> Tensor:
    r"""Pads tensor ``t`` to ``shape`` over any dimensions that need padding, with padding being
    added to the right side.

    Args:
        t:
            Tensor to pad

        shape:
            Target shape after padding

        mode:
            Passed to :func:`torch.nn.functional.pad`

        value:
            Passed to :func:`torch.nn.functional.pad`
    """
    if t.ndim != len(shape):
        raise ValueError(
            f"Number of dimensions for tensor with shape {t.shape} doesn't match padding target shape {shape}"
        )

    # F.pad has problems with zero-elem tensors, so handle this manually
    if not t.numel():
        return t.new_full(shape, value)

    delta: List[int] = []
    for i in range(t.ndim):
        real_index = -(i + 1)
        target = shape[real_index]
        current = t.shape[real_index]
        delta.append(0)
        delta.append(max(target - current, 0))
    return F.pad(t, delta, mode, value)


def padded_stack(inputs: List[Tensor], dim: int = 0, mode: str = "constant", value: float = 0) -> Tensor:
    if not inputs:
        raise ValueError("`inputs` cannot be empty")
    max_ndim = max([t.ndim for t in inputs])
    if not all([t.ndim == max_ndim for t in inputs]):
        raise ValueError("All tensors must have the same number of dimensions")

    max_size = torch.as_tensor(inputs[0].shape)
    for t in inputs:
        max_size = torch.max(max_size, torch.as_tensor(t.shape))

    target_size: List[int] = max_size.tolist()
    return torch.stack([pad(t, target_size, mode, value) for t in inputs], dim=dim)


def unpad(t: Tensor, value: float) -> Tensor:
    edges = (t != value).nonzero().amax(dim=-2)
    slices = tuple([slice(0, int(i.item()) + 1) for i in edges])
    result = t[slices]
    return result


class TensorDataclass:
    def replace(self: T, *args, **kwargs) -> T:
        return replace(self, *args, **kwargs)

    def __repr__(self) -> str:
        assert is_dataclass(self)
        s = f"{self.__class__.__name__}("
        keys = self.__dataclass_fields__.keys()  # type: ignore
        for attr_name in keys:
            attr = getattr(self, attr_name)
            if isinstance(attr, Tensor):
                if attr.numel() > 1:
                    attr_repr = repr(tuple(attr.shape))
                else:
                    attr_repr = repr(attr)
            else:
                attr_repr = repr(attr)
            s += f"{attr_name}={attr_repr}, "
        s = f"{s[:-2]})"
        return s

    def apply(self: T, dtype: Union[type, Any, Tuple[Union[type, Any]]], func: Callable, *args, **kwargs) -> T:
        return apply_to_collection(self, dtype, func, *args, **kwargs)

    def cpu(self: T) -> T:
        return move_data_to_device(self, "cpu")


U = TypeVar("U", bound="BatchMixin")


class BatchMixin:
    r"""Mixin for objects that are batched"""
    __slice_fields__: ClassVar[List[str]] = []

    @abstractproperty
    def is_batched(self) -> bool:
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...

    def __getitem__(self: U, idx: int) -> U:
        if not is_dataclass(self):
            raise NotImplementedError("BatchMixin.__getitem__ only supports dataclasses, please override __getitem__.")
        if not self.is_batched:
            raise RuntimeError("Cannot slice an unbatched object")
        if not 0 <= idx < len(self):
            raise IndexError(f"Index {idx} is invalid for object of length {len(self)}")
        if not len(self.__slice_fields__):
            raise AttributeError("Please define `__slice_fields__` to use `BatchMixin.__getitem__`")

        kwargs = {name: val for name in self.__slice_fields__ if isinstance((val := getattr(self, name)), Iterable)}
        return replace(self, **kwargs)

    @abstractclassmethod
    def from_unbatched(cls: U, examples: Iterable[U]) -> U:
        ...

    def __iter__(self: U) -> Iterator[U]:
        for pos in range(len(self)):
            yield cast(U, self[pos])

    def unbatch(self: U) -> Iterable[U]:
        if not self.is_batched:
            raise RuntimeError("This object is already unbatched")
        return cast(List[U], list(self))

    def __add__(self: U, other: U) -> U:
        unbatched = chain(self.unbatch(), other.unbatch())
        return self.from_unbatched(unbatched)

    @staticmethod
    def unpad(t: Tensor, value: float) -> Tensor:
        return unpad(t, value)

    @staticmethod
    def pad(t: Tensor, shape: List[int], mode: str = "constant", value: float = 0) -> Tensor:
        r"""Pads tensor ``t`` to ``shape`` over any dimensions that need padding, with padding being
        added to the right side.

        Args:
            t:
                Tensor to pad

            shape:
                Target shape after padding

            mode:
                Passed to :func:`torch.nn.functional.pad`

            value:
                Passed to :func:`torch.nn.functional.pad`
        """
        return pad(t, shape, mode, value)

    @staticmethod
    def padded_stack(inputs: List[Tensor], dim: int = 0, mode: str = "constant", value: float = 0) -> Tensor:
        return padded_stack(inputs, dim, mode, value)

    @staticmethod
    def requires_batched(func: Callable) -> Callable:
        def wrapper(self, *args, **kwargs):
            if not self.is_batched:
                raise RuntimeError("Object is not batched")
            return func(self, *args, **kwargs)

        return wrapper


class WandBMixin:
    @staticmethod
    def bbox_to_dict(coords: Tensor) -> Dict[str, Union[float, int]]:
        r"""Coodinate tensor to WandB dictionary entry. Assume x1, y1, x2, y2 format."""
        assert coords.numel() == 4, coords
        return {
            "minX": coords[0].item(),
            "minY": coords[1].item(),
            "maxX": coords[2].item(),
            "maxY": coords[3].item(),
        }

    @staticmethod
    def tensor_to_dict(
        tensor: Tensor, colnames: Iterable[str], lookup: Optional[Dict[Any, Any]], default: Any = None
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        assert tensor.ndim == 1
        for t, name in zip(tensor, colnames):
            val = t.item()
            result[name] = lookup.get(val, default) if lookup is not None else val
        return result

    @staticmethod
    def tensor_to_numpy(t: Tensor) -> np.ndarray:
        return t.cpu().movedim(0, -1).numpy()
