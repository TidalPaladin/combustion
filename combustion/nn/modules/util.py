#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Any, Callable, ClassVar, Dict, Final, Iterable, Optional, Tuple, Type, Union

import torch.nn as nn
import torch.nn.functional as F

from combustion.util import double, single, triple


# type hints
Kernel2D = Union[Tuple[int, int], int]
Kernel3D = Union[Tuple[int, int, int], int]
Pad2D = Union[Tuple[int, int], int]
Pad3D = Union[Tuple[int, int, int], int]
Head = Union[bool, nn.Module]

LOOKUP: Final = {"3d": 3, "2d": 2, "1d": 1}
SPATIAL_ATTRS: Final[Dict[int, Dict[str, Any]]] = {
    3: {"ToTuple": triple},
    2: {"ToTuple": double},
    1: {"ToTuple": single},
}
for module in (nn, F):
    for attr_str in dir(module):
        dim: Optional[int] = LOOKUP.get(attr_str[-2:], None)
        if dim is None:
            continue
        attr = getattr(module, attr_str)
        SPATIAL_ATTRS[dim][attr_str[:-2]] = attr


class SpatialMeta(type):
    r"""Metaclass that attaches class attributes based on the spatial identifer in
    the class name. For example:
        MyClass1d -> MyClass1d.Conv == nn.Conv1d
        MyClass3d -> MyClass3d.BatchNorm == nn.BatchNorm3d

    Attribute attachment is done by scanning ``torch.nn`` and ``torch.nn.functional`` and
    looking for classes that end in 'Nd'.
    """

    def __new__(cls: Any, name: str, bases, dct):
        x = super().__new__(cls, name, bases, dct)
        dim: int = LOOKUP.get(name[-2:].lower(), 2)
        for attr_name, attr in SPATIAL_ATTRS[dim].items():
            SpatialMeta.register_type(x, attr_name, attr)

        SpatialMeta.register_type(x, "Dim", dim, int)
        return x

    @staticmethod
    def register_type(clazz: Any, attr_name: str, attr: Any, annotation: Optional[Any] = None) -> None:
        # wrap callables as staticmethods
        if isinstance(attr, Callable) and not isinstance(attr, type):
            attr = staticmethod(attr)
        setattr(clazz, attr_name, attr)

        # try to annotate new var
        if annotation is not None:
            annotation = ClassVar[annotation]
        elif isinstance(attr, type) and issubclass(attr, nn.Module):
            annotation = ClassVar[Type[nn.Module]]
        elif isinstance(attr, Callable):
            annotation = ClassVar[Callable]
        else:
            annotation = ClassVar[Any]
        annotations: Dict[str, Any] = getattr(clazz, "__annotations__", {})
        annotations[attr_name] = annotation
        clazz.__annotations__ = annotations

    @staticmethod
    def lookup_attr(clazz: Any, choices: Iterable[Any]) -> Optional[Any]:
        name = getattr(clazz, "__name__")
        dim: int = LOOKUP.get(name[-2:], 2)
        for choice in choices:
            choice_name = getattr(choice, "__name__", None)
            if choice_name is not None and LOOKUP.get(choice_name[-2:].lower(), None) == dim:
                return choice
        return None


ConvNd = Union[Type[nn.Conv1d], Type[nn.Conv2d], Type[nn.Conv3d]]

LazyConvNd = Union[Type[nn.LazyConv1d], Type[nn.LazyConv2d], Type[nn.LazyConv3d]]

ConvTransposeNd = Union[Type[nn.ConvTranspose1d], Type[nn.ConvTranspose2d], Type[nn.ConvTranspose3d]]

LazyConvTransposeNd = Union[Type[nn.LazyConvTranspose1d], Type[nn.LazyConvTranspose2d], Type[nn.LazyConvTranspose3d]]

BatchNormNd = Union[
    Type[nn.BatchNorm1d],
    Type[nn.BatchNorm2d],
    Type[nn.BatchNorm3d],
]

AvgPoolNd = Union[Type[nn.AvgPool1d], Type[nn.AvgPool2d], Type[nn.AvgPool3d]]

MaxPoolNd = Union[Type[nn.MaxPool1d], Type[nn.MaxPool2d], Type[nn.MaxPool3d]]

AdaptiveAvgPoolNd = Union[Type[nn.AdaptiveAvgPool1d], Type[nn.AdaptiveAvgPool2d], Type[nn.AdaptiveAvgPool3d]]

AdaptiveMaxPoolNd = Union[Type[nn.AdaptiveMaxPool1d], Type[nn.AdaptiveMaxPool2d], Type[nn.AdaptiveMaxPool3d]]


class SpatialMixin:
    r"""Mixin class for SpatialMeta that provides type annotations. Static type checkers
    don't seem to find annotations that are assigned by SpatialMeta, so they are assigned
    here manually.
    """
    Conv: ClassVar[ConvNd]
    LazyConv: ClassVar[LazyConvNd]
    ConvTranspose: ClassVar[ConvTransposeNd]
    LazyConvTranspose: ClassVar[LazyConvTransposeNd]
    BatchNorm: ClassVar[BatchNormNd]
    AvgPool: ClassVar[AvgPoolNd]
    MaxPool: ClassVar[MaxPoolNd]
    AdaptiveAvgPool: ClassVar[AdaptiveAvgPoolNd]
    AdaptiveMaxPool: ClassVar[AdaptiveMaxPoolNd]
    ToTuple: Callable[[Union[int, Tuple[int, ...]]], Any]
    Dim: int
