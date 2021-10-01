#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Tuple, Union

import numpy as np
from torch import Tensor


TensorTuple = Tuple[Tensor, ...]

Array = Union[Tensor, np.ndarray]


class _MISSING:
    _instance = None

    def __hash__(self) -> int:
        return hash(repr(self))

    def __eq__(self, other) -> bool:
        return other is self._instance

    def __bool__(self) -> bool:
        return False

    def __repr__(self) -> str:
        return "???"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(_MISSING, cls).__new__(cls)
        return cls._instance

    def __add__(self, other: Any) -> Any:
        return other

    def __mul__(self, other: Any) -> Any:
        return other


MISSING: Any = _MISSING()
