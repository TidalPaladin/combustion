#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob
import itertools
import os
from abc import ABC, abstractclassmethod
from collections import OrderedDict
from itertools import islice
from typing import Callable, Generator, Iterable, Iterator, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.utils.data import Dataset

from combustion.util import Dim


try:
    import h5py
except ImportError:
    import warnings

    warnings.warn("Serialization requires h5py, please install it with" " pip install h5py")
    h5py = None


class SerializeMixin:
    """
    Mixin to enable serialization a map or iterable style dataset to disk in HDF5 file format.
    """

    def save(self, path: str, num_shards: Optional[int] = None, shard_size: Optional[int] = None) -> None:
        r"""Saves the contents of the dataset to one or more HDF5 files.

        Serialization is performed as follows:
            1.  Dataset partitions are determined if required by `num_shards` or `shard_size`. By default,
                only a single file containing the entire dataset will be produced.
            2.  Examples are read by iterating over the dataset and are written to disk. For multiple
                shards, a shard index is added to the filename given in `path`.
            3.  Attributes accessible by `vars(self)` are attached as HDF5 attributes, allowing for loading
                of instance variables. Tensors are not saved in this way, as all attributes should be small.

        .. note::
            Serialization requires the h5py library. See http://docs.h5py.org/en/stable/index.html
            for more details.

        .. note::
            When saving multiple shards, the file created at `path` will be a h5py virtual dataset.

        Args:
            path (str): The filepath to save to. Ex `foo/bar.h5`
            num_shards (optional, int): If given, `num_shards` files will be created, each 
                containing `1 / num_shards` of the dataset. Exclusive with `shard_size`. 
                Must be a positive int. 
            shard_size (optional, int): If given, multiple files will be created such that
                each file contains `shard_size` examples. Exclusive with `num_shards`.
                Must be a positive int. 
        """
        if num_shards is not None and shard_size is not None:
            raise ValueError("num_shards is incompatible with shard_size, please use one or the other")
        if num_shards is not None and num_shards <= 0:
            raise ValueError(f"num_shards must be >= 1, got {num_shards}")
        if shard_size is not None and shard_size <= 0:
            raise ValueError(f"shard_size must be >= 1, got {shard_size}")
        if shard_size is None and not hasattr(self, "__len__"):
            raise ValueError("shard_size is required for datasets with no len() method")

        # calculate num shards / shard size
        if num_shards is None and shard_size is None:
            num_shards = 1
            shard_size = len(self)
        elif num_shards is not None:
            num_shards = int(num_shards)
            shard_size = len(self) // num_shards
        elif shard_size is not None:
            shard_size = int(shard_size)
            num_shards = len(self) // shard_size

        # write shards
        files = set()
        if num_shards == 1:
            f = SerializeMixin._write_shard(path, iter(self), shard_size)
            files.add(f)
        else:
            # slice self iterator for multi-sharding
            slices = [(x * shard_size, (x + 1) * shard_size) for x in range(num_shards)]
            for shard_index, (low, high) in enumerate(slices, start=1):
                data = itertools.islice(iter(self), low, high)
                f = SerializeMixin._write_shard(path, data, shard_size, shard_index)
                files.add(f)

        self._finalize_master(path, files)
        return path

    @classmethod
    def load(cls, path):
        r"""Loads the contents of a dataset previously saved with `save()`.

        Loading is performed as follows:
            1.  The given HDF5 file is imported. For a sharded dataset this is the master file.
            2.  A dataset instance is created to hold the incoming data. 
                The `__iter__`, `__len__`, and `__getitem__` methods of the instance are patched
                to return Tensors or tuples of Tensors in an identical format and order as they were
                read during the `save()` call.
            3.  Attributes from the dataset that were saved during the `save()` call are reattached to the
                newly created instance.

        .. note::
            It is necessary to perform loading using patched methods, as the original data source has 
            been replaced by HDF5 files.

        .. note::
            De-serialization requires the h5py library. See http://docs.h5py.org/en/stable/index.html
            for more details.

        .. note::
            Any dataset using this method must provide a keyword argument only constructor. 

        Args:
            path (str): The filepath to save to. Ex `foo/bar.h5`
            num_shards (optional, int): If given, `num_shards` files will be created, each 
                containing `1 / num_shards` of the dataset. Exclusive with `shard_size`. 
                Must be a positive int. 
            shard_size (optional, int): If given, multiple files will be created such that
                each file contains `shard_size` examples. Exclusive with `num_shards`.
                Must be a positive int. 
        """
        f = h5py.File(path, "r")
        keys = f.keys()

        # TODO instead of patching, maybe make HDF5Dataset class and return that?
        # can also provide abstract classmethod to map h5py.File to new dataset instance

        def patched_iter(self):
            examples = zip([f[k] for k in keys])
            for data in examples:
                tensors = [torch.as_tensor(x) for x in data]
                yield tuple(tensors) if len(tensors) > 1 else tensors

        def patched_getitem(self, pos):
            tensors = [torch.as_tensor(f[k][pos]) for k in keys]
            return tuple(tensors) if len(tensors) > 1 else tensors

        def patched_len(self, pos):
            return len(f[next(iter(keys))])

        # patch dataset methods to return data from hdf5 files
        dataset = cls()
        setattr(dataset, "__iter__", patched_iter)
        setattr(dataset, "__getitem__", patched_getitem)
        setattr(dataset, "__len__", patched_len)

        # set attributes
        for key, value in f.attrs.items():
            setattr(dataset, key, value)

        return dataset

    @staticmethod
    def _write_shard(path, source, shard_size, shard_index=None):
        if shard_index is not None:
            path, ext = os.path.splitext(path)
            path = path + f"_{shard_index}" + ext

        with h5py.File(path, "w") as f:
            for example_index, example in enumerate(source):
                example = (example,) if isinstance(example, Tensor) else example
                for i, tensor in enumerate(example):
                    key = f"data_{i}"
                    if key not in f.keys():
                        f.create_dataset(key, (shard_size, *tensor.shape))
                    f[key][example_index, ...] = tensor

            if shard_index is not None:
                f.attrs["shard_index"] = int(shard_index)
        return path

    def _finalize_master(self, path, files):
        # create virtual dataset as master for multiple shards
        if len(files) > 1:
            first_file = next(iter(files))
            data_keys = [k for k in h5py.File(first_file, "r").keys() if "data_" in k]
            with h5py.File(path, "w") as f:
                for key in data_keys:
                    data_shape = h5py.File(first_file, "r")[key].shape
                    layout = h5py.VirtualLayout(shape=(len(files),) + data_shape)
                    for i, filename in enumerate(files):
                        vsource = h5py.VirtualSource(filename, key, shape=data_shape)
                        layout[i, ...] = vsource
                    f.create_virtual_dataset(key, layout, fillvalue=0)

        # set object attributes on master
        with h5py.File(path, "a") as f:
            for key, value in vars(self).items():
                if not isinstance(value, Tensor):
                    f.attrs[key] = value

        return path
