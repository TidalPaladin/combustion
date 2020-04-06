#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Generic, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor


TensorTuple = Tuple[Tensor, ...]

Array = Union[Tensor, np.array]
