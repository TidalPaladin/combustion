#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import glob
import itertools
import os
import warnings
from typing import Any, Callable, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.utils.data import Dataset


try:
    import h5py
except ImportError:
    warnings.warn('Serialization to HDF5 requires h5py, please install it with "pip install h5py"')
    h5py = None


def save_hdf5(
    dataset: Dataset,
    path: str,
    num_shards: Optional[int] = None,
    shard_size: Optional[int] = None,
    verbose: bool = True,
) -> None:
    r"""Saves the contents of the dataset to one or more HDF5 files.

    Serialization is performed as follows:
        1.  Dataset partitions are determined if required by ``num_shards`` or ``shard_size``. By default,
            only a single file containing the entire dataset will be produced.
        2.  Examples are read by iterating over the dataset and are written to disk. For multiple
            shards, a shard index is added to the filename given in ``path``.
        3.  Attributes accessible by ``vars(self)`` are attached as HDF5 attributes, allowing for loading
            of instance variables. Tensors are not saved in this way, as all attributes should be small.

    .. note::
        Serialization requires the h5py library. See http://docs.h5py.org/en/stable/index.html
        for more details.

    .. note::
        When saving multiple shards, the file created at ``path`` will be created from a
        :class:`h5py.VirtualSource`. See `Virtual Dataset <http://docs.h5py.org/en/stable/vds.html>`_
        for more details.

    Args:
        dataset (Datset): The dataset to save.
        path (str): The filepath to save to. Ex ``foo/bar.h5``.
        num_shards (int, optional): If given, `num_shards` files will be created, each
            containing ``1 / num_shards`` of the dataset. Exclusive with ``shard_size``.
            Must be a positive int.
        shard_size (int, optional): If given, multiple files will be created such that
            each file contains ``shard_size`` examples. Exclusive with ``num_shards``.
            Must be a positive int.
        verbose (bool, optional): If False, do not print progress updates during saving.
    """
    if num_shards is not None and shard_size is not None:
        raise ValueError("num_shards is incompatible with shard_size, please use one or the other")
    if num_shards is not None and num_shards <= 0:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")
    if shard_size is not None and shard_size <= 0:
        raise ValueError(f"shard_size must be >= 1, got {shard_size}")
    if shard_size is None and not hasattr(dataset, "__len__"):
        raise ValueError("shard_size is required for datasets with no len() method")

    # calculate num shards / shard size
    if num_shards is None and shard_size is None:
        num_shards = 1
        shard_size = len(dataset)
    elif num_shards is not None:
        num_shards = int(num_shards)
        shard_size = len(dataset) // num_shards
    elif shard_size is not None:
        shard_size = int(shard_size)
        num_shards = len(dataset) // shard_size

    # write shards
    files = set()
    if num_shards == 1:
        f = _write_shard(path, iter(dataset), shard_size, verbose=verbose)
        files.add(f)
    else:
        # slice dataset iterator for multi-sharding
        slices = [(x * shard_size, (x + 1) * shard_size) for x in range(num_shards)]
        for shard_index, (low, high) in enumerate(slices, start=1):
            data = itertools.islice(iter(dataset), low, high)
            f = _write_shard(path, data, shard_size, shard_index, verbose=verbose)
            files.add(f)

    _finalize_master(dataset, path, files)
    return path


def save_torch(dataset: Dataset, path: str, prefix: str = "example_", verbose: bool = True) -> None:
    r"""Saves the contents of the dataset to multiple files using :func:`torch.save`.

    .. note::
        This is less elegant than HDF5 serialization, but is a thread safe alternative.

    Args:
        dataset (Dataset): The dataset to save.
        path (str): The filepath to save to. Ex ``foo/bar``.
        prefix (str, optional): A prefix to append to each ``.pth`` file. Output files will be of
            the form ``{path}/{prefix}{index}.pth``
        verbose (bool, optional): If False, do not print progress updates during saving.
    """
    if not os.path.exists(path):
        os.mkdir(path)

    if verbose:
        print(f"Writing to {path}", end="", flush=True)

    for i, example in enumerate(dataset):
        target = os.path.join(path, f"{prefix}{i}.pth")
        torch.save(example, target)
        if verbose:
            print(".", end="", flush=True)


class SerializeMixin:
    r"""Mixin to enable serialization a map or iterable style dataset to disk in
    HDF5 or Torch file format.
    """

    def save(
        self,
        path: str,
        fmt: str = "hdf5",
        num_shards: Optional[int] = None,
        shard_size: Optional[int] = None,
        prefix: str = "example_",
        verbose: bool = True,
    ) -> None:
        r"""Saves the contents of the dataset to disk. See :func:`save_hdf5` and :func:`save_torch` respectively for more information
        on how saving functions for HDF5 or Torch files.

        .. note::
            Serialization requires the h5py library. See http://docs.h5py.org/en/stable/index.html
            for more details.

        Args:
            path (str): The filepath to save to. Ex `foo/bar.h5`
            fmt (str, optional): The format to save in. Should be one of ``hdf5``, ``torch``.
            num_shards (int, optional): If given, `num_shards` files will be created, each
                containing ``1 / num_shards`` of the dataset. Exclusive with ``shard_size``.
                Must be a positive int. Only has an effect when ``fmt`` is ``"hdf5"``.
            shard_size (int, optional): If given, multiple files will be created such that
                each file contains ``shard_size`` examples. Exclusive with ``num_shards``.
                Must be a positive int. Only has an effect when ``fmt`` is ``"hdf5"``.
            prefix (str, optional): Passted to :func:`save_torch` if ``fmt`` is ``"hdf5"``
            verbose (bool, optional): If False, do not print progress updates during saving.
        """
        if fmt == "hdf5":
            return save_hdf5(self, path=path, num_shards=num_shards, shard_size=shard_size, verbose=verbose)
        elif fmt == "torch":
            return save_torch(self, path=path, prefix=prefix, verbose=verbose)
        else:
            raise ValueError(f"Expected fmt to be one of 'hdf5', 'torch': found {fmt}")

    @staticmethod
    def load(
        path: str,
        fmt: Optional[str] = None,
        transform: Optional[Callable[[Tensor], Any]] = None,
        target_transform: Optional[Callable[[Tensor], Any]] = None,
    ) -> HDF5Dataset:
        r"""
        Loads the contents of a dataset previously saved with `save()`, returning a :class:`HDF5Dataset`.

        .. warning::
            Using HDF5 in a parallel / multithreaded manner poses additional challenges that have
            not yet been overcome. As such, using a :class:`HDF5Dataset` with
            :class:`torch.utils.data.DataLoader` when ``num_workers > 1`` will yield incorrect data.
            For in situations where multiple threads will be used, prefer saving with ``fmt="torch"``.
            See `Parallel HDF5 <http://docs.h5py.org/en/stable/mpi.html>`_ for more details.

        .. note::
            Loading HDF5 files requires the h5py library. See http://docs.h5py.org/en/stable/index.html
            for more details.

        .. note::
            Dataset attributes are preserved when loading a HDF5 file, but not a Torch file.

        Args:
            path (str): The filepath to load from. See `HDF5Dataset.load()` for more details
            fmt (str, optional): The expected type of data to load. By default the data type is inferred
                from the file extensions found in ``path``. HDF5 files are matched by the ``.h5`` extension,
                and Torch files are matched by the ``.pth`` extension.
                If a mix of ``hdf5`` and ``pth`` files are present in ``path``, ``fmt`` can be used
                to ensure only the desired file types are loaded.
            transform (callable, optional): A tranform to be applied to the data tensor
                See `HDF5Dataset` for more details
            target_transform (callable, optional): A tranform to be applied to the label tensor
                See `HDF5Dataset` for more details
        """
        pth_pattern = os.path.join(path, "*.pth")

        # respect user choice of fmt
        if fmt == "hdf5":
            return HDF5Dataset(path, transform, target_transform)
        elif fmt == "torch":
            return TorchDataset(path, transform, target_transform)

        # try hdf5 first if present, then try torch
        elif ".h5" in str(path) or "hdf5" in str(path):
            return HDF5Dataset(path, transform, target_transform)
        elif list(glob.glob(pth_pattern)):
            return TorchDataset(path, transform, target_transform)

        else:
            raise FileNotFoundError(f"Could not find a target to load in path {path}")


class HDF5Dataset(Dataset, SerializeMixin):
    r"""Dataset used to read from HDF5 files. See :class:`SerializeMixin` for more details

    .. note::
        Requires the h5py library. See http://docs.h5py.org/en/stable/index.html
        for more details.

    .. note::
        This class is intended for use with HDF5 files produced by Combustion's save methods.
        It may work with other HDF5 files, but this has not been verified yet.

    Args:
        path (str): The filepath to load from. When loading a sharded dataset, `path` should
            point to the virtual dataset master file. Ex ``"foo/bar.h5"``
        transform (optional, callable): Transform to be applied to data tensors.
        target_transform (optional, callable): Transform to be applied to label tensors. If
            given, the loaded dataset must produce
    """

    def __init__(
        self,
        path: str,
        transform: Optional[Callable[[Tensor], Any]] = None,
        target_transform: Optional[Callable[[Tensor], Any]] = None,
    ):

        # ensure private vars to avoid conflicts when loading keys from dataset
        self._hdf5_file = h5py.File(path, "r")
        self._keys = self._hdf5_file.keys()
        self._transform = transform
        self._target_transform = target_transform

        # set attributes that were attached to serialized dataset
        for key, value in self._hdf5_file.attrs.items():
            setattr(self, key, value)

    def __repr__(self):
        rep = f"HDF5Dataset({self._hdf5_file}, keys={list(self._keys)}, len={len(self)}"
        if self._transform is not None:
            rep += f", transform={self._transform}"
        if self._target_transform is not None:
            rep += f", transform={self._target_transform}"
        rep += ")"
        return rep

    def __getitem__(self, pos: int) -> Union[Tensor, Tuple[Tensor, ...]]:
        tensors = [torch.from_numpy(self._hdf5_file[k][pos]) for k in self._keys]
        return self.__postprocess(tensors)

    def __len__(self):
        lengths = [len(self._hdf5_file[k]) for k in self._keys]
        assert len(set(lengths)) == 1, "all lengths equal"
        return lengths[0]

    def __postprocess(self, tensors: List[Tensor]) -> Union[Tensor, Tuple[Tensor, ...]]:
        if len(tensors) < 0:
            raise RuntimeError("Loaded dataset returned no tensors")

        # require two or more tensors when target transform given
        if self._target_transform is not None and len(tensors) < 2:
            raise RuntimeError(
                "Expected loaded dataset to return 2 tensors" f"when target_transform is given, found {len(tensors)}"
            )

        # warn if more than 2 tensors - result will be
        #    (transform(t1), target_transform(t2), t3, ...)
        if (self._transform is not None or self._target_transform is not None) and len(tensors) > 2:
            warnings.warn(
                f"Loaded dataset returned {len(tensors)} tensors when transform/target_transform "
                "given. Only tensors 1 and 2 will have a transform applied."
            )

        if self._transform is not None:
            tensors[0] = self._transform(tensors[0])
        if self._target_transform is not None:
            tensors[1] = self._target_transform(tensors[1])

        return tuple(tensors) if len(tensors) > 1 else tensors[0]


class TorchDataset(Dataset, SerializeMixin):
    r"""Dataset used to read serialized examples in torch format. See :class:`SerializeMixin` for more details.

    Args:
        path (str): The path to the saved dataset. Note that unlike :class:`HDF5Dataset`, ``path``
            is a directory rather than a file.
        transform (optional, callable): Transform to be applied to data tensors.
        target_transform (optional, callable): Transform to be applied to label tensors. If
            given, the loaded dataset must produce
    """

    def __init__(
        self,
        path: str,
        transform: Optional[Callable[[Tensor], Any]] = None,
        target_transform: Optional[Callable[[Tensor], Any]] = None,
    ):
        pattern = os.path.join(path, "*.pth")
        self.path = path
        self.files = sorted(list(glob.glob(pattern)))
        self._transform = transform
        self._target_transform = target_transform

    def __repr__(self):
        rep = f"TorchDataset({self.path}"
        if self._transform is not None:
            rep += f", transform={self._transform}"
        if self._target_transform is not None:
            rep += f", transform={self._target_transform}"
        rep += ")"
        return rep

    def __getitem__(self, pos: int) -> Union[Tensor, Tuple[Tensor, ...]]:
        target = self.files[pos]
        example = torch.load(target, map_location="cpu")
        return self.__postprocess(list(example))

    def __len__(self):
        return len(self.files)

    def __postprocess(self, tensors: List[Tensor]) -> Union[Tensor, Tuple[Tensor, ...]]:
        if len(tensors) < 0:
            raise RuntimeError("Loaded dataset returned no tensors")

        # require two or more tensors when target transform given
        if self._target_transform is not None and len(tensors) < 2:
            raise RuntimeError(
                "Expected loaded dataset to return 2 tensors" f"when target_transform is given, found {len(tensors)}"
            )

        # warn if more than 2 tensors - result will be
        #    (transform(t1), target_transform(t2), t3, ...)
        if (self._transform is not None or self._target_transform is not None) and len(tensors) > 2:
            warnings.warn(
                f"Loaded dataset returned {len(tensors)} tensors when transform/target_transform "
                "given. Only tensors 1 and 2 will have a transform applied."
            )

        if self._transform is not None:
            tensors[0] = self._transform(tensors[0])
        if self._target_transform is not None:
            tensors[1] = self._target_transform(tensors[1])

        return tuple(tensors) if len(tensors) > 1 else tensors[0]


def _write_shard(path, source, shard_size, shard_index=None, verbose=True):
    if shard_index is not None:
        path, ext = os.path.splitext(path)
        path = path + f"_{shard_index}" + ext

    if verbose:
        print(f"Writing file {path}", end="", flush=True)

    with h5py.File(path, "w") as f:
        for example_index, example in enumerate(source):
            example = (example,) if isinstance(example, Tensor) else example
            for i, tensor in enumerate(example):
                key = f"data_{i}"
                if key not in f.keys():
                    f.create_dataset(key, (shard_size, *tensor.shape))
                f[key][example_index, ...] = tensor

            if verbose:
                print(".", end="", flush=True)

        if shard_index is not None:
            f.attrs["shard_index"] = int(shard_index)
    return path


def _finalize_master(dataset, path, files):
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
        for key, value in vars(dataset).items():
            if not isinstance(value, Tensor):
                try:
                    f.attrs[key] = value
                except TypeError:
                    pass

    return path


__all__ = ["save_hdf5", "save_torch", "SerializeMixin", "HDF5Dataset", "TorchDataset"]
