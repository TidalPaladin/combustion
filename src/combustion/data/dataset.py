#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import os
import random
from argparse import Namespace
from functools import partial
from itertools import islice
from pathlib import Path
from typing import Callable, Generator, Iterable, List, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.utils.data as data
from torch import Tensor

from .batch import Batch
from .preprocessing import preprocess
from .window import Window


class MatlabDataset(data.Dataset):
    """MatlabDataset
    Handler for data stored in hdf5 format as `.mat` files.
    """

    def __init__(
        self,
        path: str,
        data_key: str,
        label_key: str,
        transform: Optional[Callable[[Tensor, Tensor], Tuple[Tensor, Tensor]]] = None,
        window: Optional[Window] = None,
        cache: bool = True,
    ):
        """__init__

        :param path: Filepath of the `.mat` file
        :type path: str
        :param data_key: Dictionary key of the frames tensor
        :type data_key: str
        :param str: Dictionary key of the labels tensor
        :param transform: An optional postprocessing transform to apply to the frames/labels
        :type transform: Optional[Callable[[Tensor, Tensor], Tuple[Tensor, Tensor]]] 
        :param window: An optional Window instance to apply to the raw frame, label tuples
        :type window: Optional[Window]
        """
        super(MatlabDataset, self).__init__()
        self.file = h5py.File(path, "r")
        self.data_key = data_key
        self.label_key = label_key
        self.transform = transform
        self.window = window

        self.cache = cache
        if cache:
            self._data = np.asarray(self.file[self.data_key])
            self._labels = np.asarray(self.file[self.label_key])
        else:
            self._data = None
            self._labels = None

    def __len__(self) -> int:
        """__len__
        The number of examples in the dataset

        :rtype: int
        """
        length = len(self.frames)
        if self.window is not None:
            length -= len(self.window) - 1
        return length

    @property
    def frames(self) -> h5py.Dataset:
        """frames
        Gets the raw hdf5 frame data from disk

        :rtype: h5py.Dataset
        """
        return self._data if self._data is not None else self.file[self.data_key]

    @property
    def labels(self) -> h5py.Dataset:
        """labels
        Gets the raw hdf5 label data from disk

        :rtype: h5py.Dataset
        """
        return self._labels if self._labels is not None else self.file[self.label_key]

    def __getitem__(self, pos: int) -> Tuple[Tensor, Tensor]:
        """__getitem__
        Gets the `pos`th example from the dataset. If a window / preprocessing transform
        were specified, these will be applied before returning the final result.

        :param pos: The index of the example to retrieve
        :type pos: int
        :rtype: Tuple[Tensor, Tensor]
        """
        # map pos to one or more indices based on window choice
        if self.window is not None:
            pos += self.window.before
            indices = list(self.window.indices(pos))
        else:
            indices = [pos]

        if indices[0] < 0 or indices[-1] >= len(self.frames):
            raise IndexError("indices %s out of bounds for dataset of length %d" % (str(indices), len(self.frames)))

        frames = self.frames[indices]
        label = self.labels[pos]

        if frames.dtype == np.uint16:
            frames = frames.astype(np.int16)
        if label.dtype == np.uint16:
            label = label.astype(np.uint8)

        frames = torch.as_tensor(frames).refine_names("D", "H", "W")
        label = torch.as_tensor(label).refine_names("H", "W")
        return self.__postprocess(frames, label)

    def __iter__(self) -> Generator[Tuple[Tensor, Tensor], None, None]:
        """__iter__
        Iterates over examples in the dataset.

        :rtype: Generator[Tuple[Tensor, Tensor],None,None]
        """
        examples = zip(self.frames, self.labels)
        depth = 0
        if self.window is None:
            for frame, label in examples:
                frame = torch.as_tensor(frame).refine_names("H", "W")
                label = torch.as_tensor(label).refine_names("H", "W")
                yield self.__postprocess(frame, label)
        else:
            for frame, label in self.window(examples):
                frame = torch.as_tensor(frame)
                label = torch.as_tensor(label)
                yield self.__postprocess(frame, label)

    def __postprocess(self, frames: Tensor, label: Tensor) -> Tuple[Tensor, Tensor]:
        full_names = ("D", "C", "H", "W")

        # apply names needed for transform func
        frames = frames.align_to(*full_names)
        label = label.align_to(*full_names)
        if self.transform is not None:
            frames, label = self.transform(frames=frames, labels=label)

        # reapply final names that were dropped by transform func
        frames = frames.refine_names(*full_names).align_to("C", "D", "H", "W")
        label = label.refine_names(*full_names).align_to("C", "D", "H", "W")

        frames.requires_grad_(True)
        return frames, label

    @classmethod
    def from_args(cls, args: Namespace, filename: str) -> "MatlabDataset":
        """from_args
        Creates a MatlabDataset based on CLI flags. A filename is required to allow
        for reading of multiple source files in a single directory.

        :param args: The CLI flags
        :type args: Namespace
        :param filename: The filename to load
        :type filename: str
        """
        r = random.Random()
        r.seed(args.seed)

        def transform(frames, labels):
            return preprocess(frames=frames, labels=labels, args=args, seed=r.randint(1, 1e6))

        if args.dense_window or args.sparse_window:
            window = Window.from_args(args)
        else:
            window = None
        return cls(filename, args.matlab_data_key, args.matlab_label_key, transform=transform, window=window)


class MBBatch(Batch):
    """MBBatch
    Class representing batches of training examples. The collate_fn method is used by
    torch.utils.data.DataLoader to convert a list of examples into a single batch.
    """

    _frames = None
    _labels = None

    def __init__(self, frames, labels, **kwargs):
        super(MBBatch, self).__init__(frames=frames, labels=labels, **kwargs)

    @property
    def frames(self):
        return self._frames

    @property
    def labels(self):
        return self._labels

    @classmethod
    def collate_fn(cls, examples) -> "MBBatch":
        """collate_fn

        :param batch:
        :type batch: List[Tuple[Tensor, Tensor]]
        :rtype: MBBatch
        """
        """collate_fn
        Collates a list of examples to create a batch

        :param batch:
        """
        frames_list, labels_list = list(zip(*examples))
        frames_list = [f.rename(None) for f in frames_list]
        labels_list = [f.rename(None) for f in labels_list]
        frames = torch.stack(frames_list, 0)
        labels = torch.stack(labels_list, 0).squeeze(-3)
        return cls(frames=frames, labels=labels)
