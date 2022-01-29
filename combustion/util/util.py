#!/usr/bin/env python
# -*- coding: utf-8 -*-

import typing
from enum import IntEnum
from typing import Any, Callable, Iterable, Sized, Tuple, TypeVar, Union

from decorator import decorator


T = TypeVar("T")


class Dim(IntEnum):
    BATCH = 0
    CHANNEL = 1
    DEPTH = -3
    HEIGHT = -2
    WIDTH = -1


def arg_factory(*required_args):
    def caller(f, cls, args, *pos, **kw):
        for item in required_args:
            if not hasattr(args, item):
                raise ValueError(f"expected {item} in argparse args")
        return f(cls, args, *pos, **kw)

    return classmethod(decorator(caller))


def one_diff_tuple(count: int, default: Any, new: Any, pos: int) -> Tuple[Any, ...]:
    result = [default] * count
    result[pos] = new
    return tuple(result)


def replace_tuple(tup, pos, new):
    return tup[:pos] + (new,) + tup[pos + 1 :]


def ntuple(count: int) -> Callable[[Union[T, Iterable[T]]], Tuple[T, ...]]:
    r"""Returns a function that will accept either a single value or tuple of length ``count``
    and return a tuple of length ``count``.

    Args:
        count (int):
            Size of the output tuple

    Example:
        >>> twotuple = ntuple(2)
        >>> twotuple(1) # (1, 1)
        >>> twotuple((3, 3)) # (3, 3)
    """

    def func(arg: Union[T, Iterable[T]]) -> Tuple[T, ...]:
        if not isinstance(arg, Iterable):
            return (arg,) * count
        if isinstance(arg, Sized) and len(arg) != count:
            raise ValueError(f"expected {count}-tuple but found {arg}")
        arg = typing.cast(Iterable, arg)
        return tuple(arg)

    return func


def single(arg: Union[T, Iterable[T]]) -> Tuple[T]:
    return typing.cast(Tuple[T], _single(arg))


def double(arg: Union[T, Iterable[T]]) -> Tuple[T, T]:
    return typing.cast(Tuple[T, T], _double(arg))


def triple(arg: Union[T, Iterable[T]]) -> Tuple[T, T, T]:
    return typing.cast(Tuple[T, T, T], _triple(arg))


_single = ntuple(1)
_double = ntuple(2)
_triple = ntuple(3)
