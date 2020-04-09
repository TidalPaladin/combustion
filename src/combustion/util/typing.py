#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Tuple, Union

import numpy as np
from torch import Tensor


TensorTuple = Tuple[Tensor, ...]

Array = Union[Tensor, np.array]
