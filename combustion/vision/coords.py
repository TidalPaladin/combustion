#!/usr/bin/env python
# -*- coding: utf-8 -*-


from dataclasses import dataclass, replace
from functools import cached_property
from typing import Any, Dict, Optional, Sequence, Tuple, Type, TypeVar, Union, Iterable, cast

import torch
from torch import Tensor
from ..util.dataclasses import BatchMixin, TensorDataclass


Version = str
MaybeTargetDict = Tuple[Dict[str, Any], Optional[Dict[str, Any]]]


U = TypeVar("U", bound="Coordinates")


def _prep_torch_function_arg(
    a: Union["Coordinates", Tensor],
    proto: Tensor,
) -> Tensor:
    shape = proto.shape
    if isinstance(a, Coordinates):
        return a.coords.values()
    elif isinstance(a, Tensor):
        return a.values() if a.is_sparse else a.broadcast_to(shape).sparse_mask(proto).values()
    else:
        raise TypeError(f"Expected `Coordinates` or `Tensor`, found {type(a)}")


@dataclass(frozen=True, eq=False, repr=False)
class Coordinates(TensorDataclass, BatchMixin):
    r"""Base class for coordinates, supporting conversion to various representations.
    It is expected that the underlying raw coordinates (``coords``) are stored as XY coordinate
    pairs. This container stores the underlying coordinates in a sparse tensor format.

    .. note:
        This object implements the `__torch_function__` protocol. Dense values are extracted
        from sparse input tensors and the desired operation is then performed.
        An sparse output is then created with the type, size, and indexing of the leftmost operand.
    """
    coords: Tensor

    __slice_fields__ = ["coords"]

    def __post_init__(self):
        if not isinstance(self.coords, Tensor):
            raise TypeError(f"Expected `coords` to be Tensor, found {type(self.coords)}")
        assert 2 <= self.coords.ndim <= 4

        device = self.coords.device
        for attr_name in self.__slice_fields__:
            tensor = getattr(self, attr_name)
            if isinstance(tensor, Tensor):
                assert tensor.is_sparse
                assert tensor.is_coalesced()
                assert tensor.device == device
            elif tensor is not None:
                raise TypeError(f"Expected `{attr_name}` to be Tensor or None, found {type(tensor)}")

    @property
    def shape(self) -> torch.Size:
        return self.coords.shape

    @property
    def ndim(self) -> int:
        return self.coords.ndim

    def size(self) -> torch.Size:
        return self.coords.size()

    @property
    def is_batched(self) -> bool:
        return self.coords.ndim == 4

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Coordinates) and hash(self) == hash(other)

    @cached_property
    def _hash(self) -> int:
        hash_val = 0
        for attr_name in self.__slice_fields__:
            tensor = getattr(self, attr_name)
            if isinstance(tensor, Tensor):
                indices = hash(tensor.indices().cpu().detach().numpy().tobytes())
                coords = hash(tensor.values().cpu().detach().numpy().tobytes())
                hash_val += indices + coords
        return hash_val

    def __hash__(self) -> int:
        return self._hash

    @property
    def dtype(self) -> torch.dtype:
        return self.coords.dtype

    @cached_property
    def is_fractional(self) -> bool:
        return bool(((self.coords.values() >= 0) & (self.coords.values() <= 1.0)).all())

    def to_fractional(self: U, size: Sequence[int]) -> U:
        if self.is_fractional:
            return self
        scale = self.coords.new_tensor(size)
        return self.update_values(self.coords.values() / scale)

    def from_fractional(self: U, size: Sequence[int]) -> U:
        if not self.is_fractional:
            return self
        scale = self.coords.new_tensor(size)
        return self.update_values(self.coords.values() * scale)

    def replace(self: U, **kwargs) -> U:
        if "coords" in kwargs:
            kwargs["coords"] = kwargs["coords"].coalesce()
        return replace(self, **kwargs)

    def clip_to_size(self: U, size: Sequence[int]) -> U:
        max = self.coords.new_tensor(size)
        min = torch.zeros_like(max)
        values = self.coords.values().clamp(min=min, max=max)
        return self.update_values(values)

    @classmethod
    def from_padded(cls: Type[U], coords: Tensor, pad_val: Any = 0, **tensors) -> U:
        assert 3 <= coords.ndim <= 4
        keep = (coords == pad_val).all(dim=-1).logical_not_()
        indices = (keep).nonzero().t()
        sparse_coords = torch.sparse_coo_tensor(indices, coords[keep], coords.shape, device=coords.device).coalesce()
        sparse_tensors = {
            name: torch.sparse_coo_tensor(indices, tensor[keep], tensor.shape, device=tensor.device).coalesce()
            for name, tensor in tensors.items() if isinstance(tensor, Tensor)
        }
        return cls(sparse_coords, **sparse_tensors)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        base = args[0]
        proto = args[0].coords
        args = [_prep_torch_function_arg(a, proto) for a in args]
        new_coords = func(*args, **kwargs)
        new_coords = torch.sparse_coo_tensor(proto.indices(), new_coords, proto.shape, device=proto.device)
        return base.replace(coords=new_coords.coalesce())

    def sparse_mask(self, data: Tensor) -> Tensor:
        if data.is_sparse:
            data = data.to_dense()
        return data.broadcast_to(self.size()).sparse_mask(self.coords)

    def update_values(self: U, data: Tensor) -> U:
        assert data.shape == self.coords.values().shape
        new_coords = torch.sparse_coo_tensor(self.coords.indices(), data, self.shape, device=self.coords.device)
        return self.replace(coords=new_coords.coalesce())

    def amin(self, *args, **kwargs) -> Tensor:
        dense = torch.full(self.coords.shape, float("inf"), device=self.coords.device)
        indexor = [t.view(-1) for t in self.coords.indices().t().split(1, dim=-1)]
        dense[indexor] = self.coords.values()
        result = dense.amin(*args, **kwargs)
        result[result == float("inf")] = float("nan")
        return result

    def amax(self, *args, **kwargs) -> Tensor:
        dense = torch.full(self.coords.shape, float("-inf"), device=self.coords.device)
        indexor = [t.view(-1) for t in self.coords.indices().t().split(1, dim=-1)]
        dense[indexor] = self.coords.values()
        result = dense.amax(*args, **kwargs)
        result[result == float("-inf")] = float("nan")
        return result

    @cached_property
    def box(self) -> "BoundingBox2d":
        mins = self.amin(dim=-2)
        maxes = self.amax(dim=-2)
        coords = torch.cat([mins, maxes], dim=-1)
        return BoundingBox2d.from_xyxy(coords, pad_val=float("nan"))

    @classmethod
    def from_unbatched(cls: Type[U], examples: Iterable[U]) -> U:
        assert not any(t.is_batched for t in examples)
        kwargs: Dict[str, Tensor] = {}
        for attr_name in cls.__slice_fields__:
            values = [getattr(e, attr_name) for e in examples]
            if all(isinstance(v, Tensor) for v in values):
                kwargs[attr_name] = torch.stack(values).coalesce()
            elif any(isinstance(v, Tensor) for v in values):
                raise ValueError(f"Found a mix of tensors and other type for attribute {attr_name}")
        return cls(**kwargs)


@dataclass(frozen=True, repr=False, eq=False)
class BoundingBox2d(Coordinates):
    def __post_init__(self):
        assert self.coords.shape[-1] == 2

    def __hash__(self) -> int:
        return self._hash

    @classmethod
    def from_xyxy(cls, box: Tensor, pad_val: Any = 0) -> "BoundingBox2d":
        x1, y1, x2, y2 = torch.split(box, 1, dim=-1)
        coords = torch.stack(
            [
                torch.cat([x1, y1], dim=-1),
                torch.cat([x2, y1], dim=-1),
                torch.cat([x2, y2], dim=-1),
                torch.cat([x1, y2], dim=-1),
            ],
            dim=-2,
        )
        return cls.from_padded(coords, pad_val)

    @cached_property
    def xyxy(self) -> Tensor:
        x1y1 = self.coords[..., 0, :]
        x2y2 = self.coords[..., 2, :]
        return torch.cat([x1y1, x2y2], dim=-1)

    @property
    def area(self) -> Tensor:
        _, _, w, h = self.xywh
        return w * h

    @property
    def yxyx(self) -> Tensor:
        return self.xyxy.flip(-1).roll(2, dims=-1)

    @property
    def xywh(self) -> Tensor:
        mins = self.xyxy[..., :2]
        maxes = self.xyxy[..., 2:]
        xy = (mins + maxes) / 2
        wh = maxes - mins
        return torch.cat([xy, wh], dim=-1)

    @classmethod
    def from_yxyx(cls, box: Tensor, pad_val: Any = 0) -> "BoundingBox2d":
        box = box.flip(-1).roll(2, dims=-1)
        return cls.from_xyxy(box, pad_val)
