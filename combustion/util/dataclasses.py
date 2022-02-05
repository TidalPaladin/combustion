#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABC, abstractclassmethod, abstractmethod, abstractproperty
from dataclasses import is_dataclass, replace
from itertools import chain
from typing import Any, Callable, ClassVar, Dict, Iterable, Iterator, List, Optional, Set, Tuple, TypeVar, Union, cast

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


def pretty_repr(attr: Tensor) -> Union[str, Tuple[int, ...]]:
    if attr.numel() > 1:
        return tuple(attr.shape)
    else:
        return repr(attr)


class TensorDataclass:
    def replace(self: T, *args, **kwargs) -> T:
        return replace(self, *args, **kwargs)

    def __repr__(self) -> str:
        assert is_dataclass(self)
        repr_dc = apply_to_collection(self, Tensor, pretty_repr)
        s = f"{self.__class__.__name__}(\n"
        keys = self.__dataclass_fields__.keys()  # type: ignore
        indent = "    "
        for attr_name in keys:
            attr = getattr(repr_dc, attr_name)
            attr_repr = str(attr).replace("\n", f"\n{indent}")
            s += f"{indent}{attr_name}={attr_repr}, \n"
        s += ")"
        return s

    def apply(self: T, dtype: Union[type, Any, Tuple[Union[type, Any]]], func: Callable, *args, **kwargs) -> T:
        return apply_to_collection(self, dtype, func, *args, **kwargs)

    def cpu(self: T) -> T:
        return move_data_to_device(self, "cpu")

    def to(self: T, *args, **kwargs) -> T:
        return self.apply(Tensor, Tensor.to, *args, **kwargs)

    def clone(self: T, *args, **kwargs) -> T:
        return self.apply(Tensor, Tensor.clone, *args, **kwargs)

    def detach(self: T, *args, **kwargs) -> T:
        return self.apply(Tensor, Tensor.detach, *args, **kwargs)

    @property
    def device(self) -> Optional[torch.device]:
        seen_devices: Set[torch.device] = set()

        def check_device(x: Tensor) -> Tensor:
            seen_devices.add(x.device)
            return x

        self.apply(Tensor, check_device)
        if len(seen_devices) == 1:
            return next(iter(seen_devices))
        else:
            return None

    @property
    def requires_grad(self) -> bool:
        requires_grad: Set[bool] = set()

        def check_requires_grad(x: Tensor) -> Tensor:
            requires_grad.add(x.requires_grad)
            return x

        self.apply(Tensor, check_requires_grad)
        return any(requires_grad)


U = TypeVar("U", bound="BatchMixin")


def slice_attr(attr: Any, idx: int, target_len: int) -> Any:
    # slice tensors directly
    if isinstance(attr, BatchMixin) and len(attr) == target_len:
        return attr[idx]
    elif isinstance(attr, Tensor) and attr.ndim and len(attr) == target_len:
        return attr[idx]
    # slice lists/tuples of tensors itemwise
    elif isinstance(attr, (list, tuple)):
        return attr.__class__(slice_attr(x, idx, target_len) for x in attr)
    # noop
    else:
        return attr


class BatchMixin(ABC):
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

        # if __slice_fields__ was specified, only recurse on those fields
        if len(self.__slice_fields__):
            kwargs = {
                name: slice_attr(val, idx, len(self))  # type: ignore
                for name in self.__slice_fields__
                if isinstance((val := getattr(self, name)), (Tensor, Iterable))
            }
            return replace(self, **kwargs)

        # otherwise apply recursively to all Tensor fields
        else:
            return apply_to_collection(self, Tensor, slice_attr, idx=idx, target_len=len(self))

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
