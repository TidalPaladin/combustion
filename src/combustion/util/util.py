#!/usr/bin/env python
# -*- coding: utf-8 -*-

from enum import IntEnum
from typing import Iterable, Tuple, Union

from decorator import decorator


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


def one_diff_tuple(count, default, new, pos):
    # type: (int, Union[int, float], Union[int, float], int) -> Tuple[Union[int, float], ...]
    result = [default] * count
    result[pos] = new
    return tuple(result)


def replace_tuple(tup, pos, new):
    return tup[:pos] + (new,) + tup[pos + 1 :]


def ntuple(count: int):
    def func(arg):
        if not isinstance(arg, Iterable):
            arg = (arg,) * count
        elif len(arg) != count:
            raise ValueError(f"expected {count}-tuple but found {arg}")
        return tuple(arg)

    return func


single = ntuple(1)
double = ntuple(2)
triple = ntuple(3)
