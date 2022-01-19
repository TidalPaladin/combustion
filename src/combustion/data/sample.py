#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch.utils.data import Dataset, WeightedRandomSampler, RandomSampler
from typing import Sequence, Sized, cast, Any, Iterator, Tuple
from itertools import cycle


class MixedDataset(Dataset):
    r"""Wrapper dataset that combines multiple datasets with a given sampling ratio.

    Args:
        datasets:
            Datasets to combine

        weights:
            Sampling weights for each datasets. Must have ``len(weights) == len(datasets) and sum(weights) == 1``.
    """
    def __init__(self, datasets: Sequence[Dataset], weights: Sequence[float]):
        if not len(datasets) == len(weights):
            raise ValueError(f"Number of datasets {len(datasets)} must match number of weights {len(weights)}")
        if not 0.98 <= sum(weights) <= 1.02:
            raise ValueError(f"Weights {weights} did not sum to 1")
        assert all(isinstance(x, Sized) for x in datasets)
        self.datasets = tuple(datasets)
        self.total_length = sum(len(cast(Sized, x)) for x in datasets)
        self.sampler = WeightedRandomSampler(weights, self.total_length)
        self.dataset_index_buffer = cycle(iter(self.sampler))
        self.subsamplers = tuple(cycle(iter(RandomSampler(x))) for x in self.datasets)

    def __len__(self) -> int:
        return self.total_length

    def __getitem__(self, idx: int) -> Any:
        sampled_dataset_idx, real_idx = self.get_real_indices(idx)
        dataset = self.datasets[sampled_dataset_idx]
        return dataset[real_idx]

    def get_real_indices(self, idx: int) -> Tuple[int, int]:
        sampled_dataset_idx = next(self.dataset_index_buffer)
        dataset = self.datasets[sampled_dataset_idx]
        subsampler = self.subsamplers[sampled_dataset_idx]
        real_idx = next(subsampler)
        return sampled_dataset_idx, real_idx

    @property
    def iter_indices(self) -> Iterator[Tuple[int, int]]:
        r"""Iterator of tuples giving the dataset and scaled index"""
        for i in range(len(self)):
            yield self.get_real_indices(i)
