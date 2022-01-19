#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import pytest
from combustion.data import MixedDataset
from torch.utils.data import Dataset

class DummyDataset(Dataset):
    def __init__(self, start: int, stop: int):
        self.start = start
        self.stop = stop

    def __getitem__(self, pos: int) -> int:
        if not 0 <= pos < len(self):
            raise IndexError(pos)
        return self.start + pos

    def __len__(self) -> int:
        return self.stop - self.start

@pytest.mark.parametrize("ratios", [
    [0.5, 0.5],
    [0.9, 0.1],
    [0.1, 0.9],
    pytest.param([0.1, 0.1], marks=pytest.mark.xfail(raises=ValueError)),
])
def test_mixed_dataset(ratios):
    torch.random.manual_seed(42)
    ds1 = DummyDataset(start=0, stop=1000)
    ds2 = DummyDataset(start=1000, stop=2000)

    mixed = MixedDataset([ds1, ds1], weights=ratios)
    assert len(mixed) == len(ds1) + len(ds2)
    indices = list(mixed.iter_indices)

    num_ds1 = sum(1 for item in indices if item[0] == 0)
    num_ds2 = sum(1 for item in indices if item[0] == 1)
    ds2_ratio = num_ds2 / len(mixed) 
    lower_bound = ratios[-1] * 0.9
    upper_bound = ratios[-1] * 1.1
    assert lower_bound <= ds2_ratio <= upper_bound
