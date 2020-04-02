#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABC, abstractclassmethod
from collections import OrderedDict
from itertools import islice
from typing import Callable, Generator, Iterable, List, Optional, Tuple, Iterator, Union

import torch
from torch import Tensor
from torch.utils.data import Dataset

from combustion.util import Dim

class AbstractDataset(ABC, Dataset):
    pass
