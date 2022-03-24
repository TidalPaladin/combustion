#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABC, abstractclassmethod, abstractmethod, abstractproperty
from dataclasses import is_dataclass, replace
from itertools import chain
from typing import Any, Callable, ClassVar, Dict, Iterable, Iterator, List, Optional, Tuple, TypeVar, Union, cast, Sequence, Type

import numpy as np
import torch
import torch.nn.functional as F
import math
from pytorch_lightning.utilities.apply_func import apply_to_collection, move_data_to_device
from torch import Tensor
import functools



T = TypeVar("T", bound="TensorDataclass")
S = TypeVar("S")
DEFAULT_PAD_VAL = float("nan")


def requires_batched(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        assert isinstance(self, BatchMixin), "requires_batched decorator should only be used with BatchMixin"
        if not self.is_batched:
            raise RuntimeError(f"{type(self)} is not batched")
        return func(self, *args, **kwargs)
    return wrapper


@torch.jit.script
def max_size(inputs: List[Tensor]) -> List[int]:
    r"""Find the maximum size per-dimension for a list of input tensors"""
    if not inputs:
        raise ValueError("`inputs` cannot be empty")
    max_ndim = max([t.ndim for t in inputs])
    if not all([t.ndim == max_ndim for t in inputs]):
        raise ValueError("All tensors must have the same number of dimensions")

    max_size = torch.as_tensor(inputs[0].shape)
    for t in inputs:
        max_size = torch.max(max_size, torch.as_tensor(t.shape))
    return max_size.tolist()


@torch.jit.script
def pad(t: Tensor, shape: List[int], mode: str = "constant", value: float = 0) -> Tensor:
    r"""Pads tensor ``t`` to ``shape`` over any dimensions that need padding, with padding being
    added to the end of each dimension.

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

    # compute padding needed
    input_size = torch.tensor(t.shape)
    target_size = torch.tensor(shape)
    delta = target_size.sub_(input_size).clamp_min_(0)

    # flip delta and interleave with zeros to satisfy F.pad signature (e.g. [0, W, 0, H, 0, D])
    delta = torch.stack([torch.zeros_like(delta), delta.flip(0)], dim=-1).flatten()

    delta_list: List[int] = delta.tolist()
    result =  F.pad(t, delta_list, mode, value)
    assert list(result.shape) == shape
    return result


@torch.jit.script
def padded_stack(inputs: List[Tensor], dim: int = 0, mode: str = "constant", pad_value: float = DEFAULT_PAD_VAL) -> Tensor:
    if not inputs:
        raise ValueError("`inputs` cannot be empty")
    target_size = max_size(inputs)
    return torch.stack([pad(t, target_size, mode, pad_value) for t in inputs], dim=dim)


@torch.jit.script
def sparse_stack(inputs: List[Tensor], dim: int = 0) -> Tensor:
    if not inputs:
        raise ValueError("`inputs` cannot be empty")
    if not all([i.is_sparse for i in inputs]):
        raise ValueError("`inputs` must all be sparse tensors")
    target_size = max_size(inputs)
    inputs = [i.clone().sparse_resize_(target_size, i.sparse_dim(), i.dense_dim()) for i in inputs]
    return torch.stack(inputs, dim).coalesce()


@torch.jit.script
def trim_padding(t: Tensor, pad_value: float = DEFAULT_PAD_VAL) -> Tensor:
    pad = torch.tensor(pad_value, device=t.device)
    # NOTE: comparison of x == float("nan") will always return False
    mask = t.isnan().logical_not_() if pad.isnan() else (t != pad_value)
    size: List[int] = mask.nonzero().amax(dim=-2).add_(1).tolist()
    result = t[mask].view(torch.Size(size))
    return result


@torch.jit.script
def trim_sparse(t: Tensor) -> Tensor:
    t = t.coalesce()
    bounds = t.coalesce().indices().amax(dim=-1).add_(1)
    bounds_list: List[int] = bounds.tolist() 
    bounds_list += list(t.shape[t.sparse_dim():])
    assert len(bounds_list) == t.ndim
    return torch.sparse_coo_tensor(t.indices(), t.values(), bounds_list, device=t.device).coalesce()


@torch.jit.script
def padded_split(
    tensor: Tensor, 
    dim: int = 0, 
    pad_value: float = DEFAULT_PAD_VAL,
) -> List[Tensor]:
    return [trim_padding(t, pad_value) for t in tensor.unbind(dim)]


@torch.jit.script
def sparse_split(
    tensor: Tensor, 
    dim: int = 0, 
) -> List[Tensor]:
    if tensor.is_sparse:
        raise ValueError("`tensor` must be sparse")
    return [trim_sparse(t) for t in tensor.unbind(dim)]


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

Sliceable = Union[Tensor, Sequence, "BatchMixin"]


def check_field(items: Sequence[U], field: str) -> bool:
    sliceable = [i._field_is_sliceable(field) for i in items]
    if all(sliceable):
        return True
    elif any(sliceable):
        raise ValueError(f"Found a mix of sliceable and non-sliceable inputs for field {field}")
    return False


class BatchMixin(ABC):
    r"""Mixin for objects that are batched"""
    __slice_fields__: ClassVar[List[str]] = []

    _HANDLED_FUNCTIONS = {}

    @abstractproperty
    def is_batched(self) -> bool:
        ...

    def _validate_slice_fields(self):
        if not self.is_batched:
            return
        for field in self.__slice_fields__:
            if not hasattr(self, field):
                raise AttributeError(f"Field {field} was in __slice_fields__ but {type(self)} has no such attribute")
            attr = getattr(self, field)
            if self._field_is_sliceable(field) and len(attr) != len(self):
                raise AttributeError(f"Expected length {len(self)} for field {field}, found {len(field)}")

    def _field_is_sliceable(self, field: str) -> bool:
        sliceable_types = (Tensor, Sequence, BatchMixin)
        attr = getattr(self, field)
        return  isinstance(attr, sliceable_types)

    def __len__(self) -> int:
        if not len(self.__slice_fields__):
            raise AttributeError("Please define `__slice_fields__` to use `BatchMixin.__len__`")
        field = self.__slice_fields__[0]
        return len(getattr(self, field))

    def __getitem__(self: U, idx: int) -> U:
        if not is_dataclass(self):
            raise NotImplementedError("BatchMixin.__getitem__ only supports dataclasses, please override __getitem__.")
        if not 0 <= idx < len(self):
            raise IndexError(f"Index {idx} is invalid for object of length {len(self)}")
        if not len(self.__slice_fields__):
            raise AttributeError("Please define `__slice_fields__` to use `BatchMixin.__getitem__`")

        kwargs: Dict[str, Sliceable] = {}
        for name, ftype, value in self.slice_fields(sliceable_only=True):
            sliced: Sliceable = value[idx]
            if isinstance(sliced, Tensor):
                # TODO triming padded tensors will only work when default pad value is used
                sliced = trim_sparse(sliced) if sliced.is_sparse else trim_padding(sliced)
            elif issubclass(ftype, Sequence):
                sliced = ftype([sliced])
            kwargs[name] = sliced

        return replace(self, **kwargs)

    def __iter__(self: U) -> Iterator[U]:
        for pos in range(len(self)):
            yield cast(U, self[pos])

    def slice_fields(self, sliceable_only: bool = False) -> Iterator[Tuple[str, Type[Sliceable], Sliceable]]:
        for field in self.__slice_fields__:
            attr = getattr(self, field)
            sliceable = self._field_is_sliceable(field)
            if sliceable or not sliceable_only:
                yield field, type(attr), attr

    @classmethod
    def collate(cls: Type[U], items: Sequence[U], pad_value: float = DEFAULT_PAD_VAL) -> U:
        replacement = {}
        for field, ftype, attr in items[0].slice_fields():
            if check_field(items, field):
                if issubclass(ftype, Tensor):
                    assert isinstance(attr, Tensor)
                    size = max_size([getattr(i, field) for i in items])
                    if attr.is_sparse:
                        replacement[field] =  sparse_stack([getattr(i, field) for i in items])
                    else:
                        replacement[field] = padded_stack([getattr(i, field) for i in items], pad_value=pad_value)
                elif issubclass(ftype, BatchMixin):
                    replacement[field] = ftype.collate(items, pad_value)
                elif issubclass(ftype, Sequence):
                    replacement[field] = sum([getattr(i, field) for i in items], ftype())
        return cls(**replacement)

    @classmethod
    def implements(cls, torch_function):
        """Register a torch function override"""
        @functools.wraps(torch_function)
        def decorator(func):
            cls._HANDLED_FUNCTIONS[torch_function] = func
            return func
        return decorator

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func not in cls._HANDLED_FUNCTIONS or not all(
            issubclass(t, (BatchMixin))
            for t in types
        ):
            return NotImplemented
        return cls._HANDLED_FUNCTIONS[func](*args, **kwargs)


@BatchMixin.implements(torch.cat)
def cat(items: List[U], *args, **kwargs) -> U:
    replacement = {
        field: torch.cat([getattr(i, field) for i in items], *args, **kwargs)
        for field, _, _ in items[0].slice_fields()
        if check_field(items, field)
    }
    return replace(items[0], **replacement)


@BatchMixin.implements(torch.stack)
def stack(items: List[U], *args, **kwargs) -> U:
    replacement = {}
    for field, _, _ in items[0].slice_fields():
        if check_field(items, field):
            if isinstance(getattr(items[0], field), (Tensor, BatchMixin)):
                replacement[field] = torch.stack([getattr(i, field) for i in items], *args, **kwargs)
            else:
                start = type(getattr(items[0], field))()
                replacement[field] = sum((getattr(i, field) for i in items), start)
    return replace(items[0], **replacement)


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
