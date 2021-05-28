#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import glob
import itertools
import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm


try:
    import h5py
except ImportError:
    h5py = None


def save_hdf5(
    dataset: Dataset,
    path: str,
    num_shards: Optional[int] = None,
    shard_size: Optional[int] = None,
    verbose: bool = True,
) -> None:
    r"""Saves the contents of the dataset to one or more HDF5 files.

    .. warning::
        HDF5 support in Combustion is deprecated

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
    _check_h5py()
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
        if not verbose:
            bar = bar("Saving dataset", max=len(dataset))
        else:
            bar = None

        # slice dataset iterator for multi-sharding
        slices = [(x * shard_size, (x + 1) * shard_size) for x in range(num_shards)]
        for shard_index, (low, high) in enumerate(slices, start=1):
            data = itertools.islice(iter(dataset), low, high)
            f = _write_shard(path, data, shard_size, shard_index, verbose=False)
            files.add(f)
            if bar is not None:
                bar.next()
        if bar is not None:
            bar.finish()

    _finalize_master(dataset, path, files)
    return path


def save_torch(
    dataset: Dataset,
    path: str,
    prefix: Union[str, Callable[[int, Any], str]] = "example_",
    verbose: bool = True,
    threads: int = 0,
) -> None:
    r"""Saves the contents of the dataset to multiple files using :func:`torch.save`.

    .. note::
        This is less elegant than HDF5 serialization, but is a thread safe alternative.

    Args:
        dataset (Dataset): The dataset to save.

        path (str): The filepath to save to. Ex ``foo/bar``.

        prefix (str or callable):
            Either a string prefix to append to each ``.pth`` file, or a callable
            that returns a such a string prefix given the example index and example tensors as input.
            Example indices are automatically appended to the target filepath when a string prefix is given,
            but not when a callable prefix is given.
            Output files will be of the form ``{path}/{prefix}{index}.pth``, or
            ``{path}/{prefix}.pth`` when a callable prefix is provided.

        verbose (bool, optional): If False, do not print progress updates during saving.

        threads (int): Parallel threads to use when serializing. By default, run single-threaded

    .. Example:
        >>> str_prefix = "example_"
        >>> save_torch(ds, path="root", prefix=str_prefix)
        >>> # creates files root/example_{index}.pth
        >>>
        >>> callable_prefix = lambda pos, example: f"class_{example[1].item()}/example_{pos}"
        >>> save_torch(ds, path="root", prefix=callable_prefix)
        >>> # creates files root/class_{label_id}/example_{index}.pth

    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    if hasattr(dataset, "__len__"):
        total = len(dataset)
    else:
        total = None

    def save(example, index):
        if isinstance(prefix, str):
            target = Path(path, f"{prefix}{index}.pth")
        else:
            example_prefix = prefix(index, example)
            if not isinstance(example_prefix, str):
                raise ValueError(f"Callable `prefix` must return a str, got {type(example_prefix)}")
            target = Path(path, f"{example_prefix}.pth")
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(example, target)

    def callback(f):
        f.result()
        if f.exception() is not None:
            raise f.exception()
        bar.update(1)

    bar = tqdm(desc="Saving dataset", disable=(not verbose), total=total)
    if threads > 0:
        with ThreadPoolExecutor(threads) as tp:
            for i, example in enumerate(dataset):
                f = tp.submit(save, example, i)
                f.add_done_callback(callback)
        bar.close()

    else:
        for i, example in enumerate(dataset):
            save(example, i)
            bar.update(1)
        bar.close()


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
        threads: int = 0,
    ) -> None:
        r"""Saves the contents of the dataset to disk. See :func:`save_hdf5` and :func:`save_torch`
        respectively for more information on how saving functions for HDF5 or Torch files.

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
            threads (int): Parallel threads to use when serializing. By default, run single-threaded
        """
        if fmt == "hdf5":
            return save_hdf5(self, path=path, num_shards=num_shards, shard_size=shard_size, verbose=verbose)
        elif fmt == "torch":
            return save_torch(self, path=path, prefix=prefix, verbose=verbose, threads=threads)
        else:
            raise ValueError(f"Expected fmt to be one of 'hdf5', 'torch': found {fmt}")

    @staticmethod
    def load(
        path: str,
        fmt: Optional[str] = None,
        **kwargs,
    ) -> Union[TorchDataset, HDF5Dataset]:
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
            **kwargs:  Forwarded to the constructors for :class:`HDF5Dataset` or :class:`TorchDataset`,
                depending on what dataset is constructed.
        """
        pth_pattern = os.path.join(path, "*.pth")

        # respect user choice of fmt
        if fmt == "hdf5":
            return HDF5Dataset(path, **kwargs)
        elif fmt == "torch":
            return TorchDataset(path, **kwargs)

        # try hdf5 first if present, then try torch
        elif ".h5" in str(path) or "hdf5" in str(path):
            return HDF5Dataset(path, **kwargs)
        elif list(glob.glob(pth_pattern)):
            return TorchDataset(path, **kwargs)

        else:
            raise FileNotFoundError(f"Could not find a target to load in path {path}")


class TransformableDataset(Dataset):
    r"""Base class for datasets that accept callable transforms that are applied to each example.

    Args:

        transform (optional, callable): Transform to be applied to data tensors.
        target_transform (optional, callable): Transform to be applied to label tensors. If
            given, the loaded dataset must produce
        transforms (optional, callable): Transform to be applied to the output of ``__getitem__``,
            i.e. both data and labels. The transform should accept as many positional arguments
            as ``__getitem__`` returns.


    Example with data and label transform::

        >>> def xform(data, label):
        >>>     ...
        >>>     return new_data, new_label
        >>>
        >>> ds = TransformableDataset(transforms=xform)
    """

    def __init__(
        self,
        transform: Optional[Callable[[Tensor], Any]] = None,
        target_transform: Optional[Callable[[Tensor], Any]] = None,
        transforms: Optional[Callable[[Any], Any]] = None,
    ):

        super().__init__()
        self._transform = transform
        self._target_transform = target_transform
        self._transforms = transforms

    def _transform_repr(self):
        rep = ""
        if self._transform is not None:
            rep += f", transform={self._transform}"
        if self._target_transform is not None:
            rep += f", transform={self._target_transform}"
        if self._transforms is not None:
            rep += f", transform={self._transforms}"
        return rep

    def __repr__(self):
        return f"TransformableDataset({self._transform_repr()})"

    def apply_transforms(self, tensors: Iterable[Tensor]) -> Union[Tensor, Tuple[Tensor, ...]]:
        r"""Applies transforms to an iterable of tensors

        Args:
            tensors (iterable of tensors):
                The tensors to transform. When transforming a single tensor, wrap it in an interable.
        """
        if len(tensors) < 0:
            raise RuntimeError("No tensors were present to transform")

        # require two or more tensors when target transform given
        if self._target_transform is not None and len(tensors) < 2:
            raise RuntimeError(
                f"Expected loaded dataset to return 2 tensors when target_transform is given, found {len(tensors)}"
            )

        # warn if more than 2 tensors - result will be
        #    (transform(t1), target_transform(t2), t3, ...)
        if (self._transform is not None or self._target_transform is not None) and len(tensors) > 2:
            warnings.warn(
                f"Loaded dataset returned {len(tensors)} tensors when transform/target_transform "
                "was given. Only tensors 1 and 2 will have a transform applied."
            )

        if self._transform is not None:
            tensors[0] = self._transform(tensors[0])
        if self._target_transform is not None:
            tensors[1] = self._target_transform(tensors[1])
        if self._transforms is not None:
            tensors = self._transforms(*tensors)

        return tuple(tensors) if len(tensors) > 1 else tensors[0]


class HDF5Dataset(TransformableDataset, SerializeMixin):
    r"""Dataset used to read from HDF5 files. See :class:`SerializeMixin` for more details

    .. warning::
        HDF5 support in Combustion is deprecated

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
        transforms (optional, callable): Transform to be applied to the output of ``__getitem__``,
            i.e. both data and labels. The transform should accept as many positional arguments
            as ``__getitem__`` returns.
    """

    def __init__(
        self,
        path: str,
        transform: Optional[Callable[[Tensor], Any]] = None,
        target_transform: Optional[Callable[[Tensor], Any]] = None,
        transforms: Optional[Callable[[Any], Any]] = None,
    ):
        super().__init__(transform, target_transform, transforms)
        _check_h5py()

        # ensure private vars to avoid conflicts when loading keys from dataset
        self._hdf5_file = h5py.File(path, "r")
        self._keys = self._hdf5_file.keys()

        # set attributes that were attached to serialized dataset
        for key, value in self._hdf5_file.attrs.items():
            setattr(self, key, value)

    def __repr__(self):
        rep = f"HDF5Dataset({self._hdf5_file}, keys={list(self._keys)}, len={len(self)}"
        rep += self._transform_repr()
        rep += ")"
        return rep

    def __getitem__(self, pos: int) -> Union[Tensor, Tuple[Tensor, ...]]:
        tensors = [torch.from_numpy(self._hdf5_file[k][pos]) for k in self._keys]
        return self.apply_transforms(tensors)

    def __len__(self):
        lengths = [len(self._hdf5_file[k]) for k in self._keys]
        assert len(set(lengths)) == 1, "all lengths equal"
        return lengths[0]


class TorchDataset(TransformableDataset, SerializeMixin):
    r"""Dataset used to read serialized examples in torch format. See :class:`SerializeMixin` for more details.

    Args:
        path (str): The path to the saved dataset. Note that unlike :class:`HDF5Dataset`, ``path``
            is a directory rather than a file.
        transform (optional, callable): Transform to be applied to data tensors.
        target_transform (optional, callable): Transform to be applied to label tensors. If
            given, the loaded dataset must produce
        transforms (optional, callable): Transform to be applied to the output of ``__getitem__``,
            i.e. both data and labels. The transform should accept as many positional arguments
            as ``__getitem__`` returns.

        pattern (optional, str): Pattern of filenames to match.
    """

    def __init__(
        self,
        path: str,
        length_override: Optional[int] = None,
        transform: Optional[Callable[[Tensor], Any]] = None,
        target_transform: Optional[Callable[[Tensor], Any]] = None,
        transforms: Optional[Callable[[Any], Any]] = None,
        pattern: str = "*.pth",
    ):
        super().__init__(transform, target_transform, transforms)
        self.path = path
        self.pattern = pattern
        pattern = os.path.join(path, pattern)
        self.files = sorted(list(glob.glob(pattern)))
        if length_override is not None:
            length_override = int(length_override)
            if length_override <= 0:
                raise ValueError(f"Expected length_override > 0, found {length_override}")
            self.length_override = length_override
        else:
            self.length_override = None

    def __repr__(self):
        rep = f"TorchDataset({self.path}"
        rep += self._transform_repr()
        if self.pattern != "*.pth":
            rep += f", pattern={self.pattern}"
        rep += ")"
        return rep

    def __getitem__(self, pos: int) -> Union[Tensor, Tuple[Tensor, ...]]:
        if pos < 0 or pos > len(self):
            raise IndexError(f"{pos}")
        if self.length_override is not None:
            pos = pos % len(self.files)
        target = self.files[pos]
        example = torch.load(target, map_location="cpu")
        return self.apply_transforms(list(example))

    def __len__(self):
        if self.length_override is not None:
            return self.length_override
        else:
            return len(self.files)


def _write_shard(path, source, shard_size, shard_index=None, verbose=True):
    if shard_index is not None:
        path, ext = os.path.splitext(path)
        path = path + f"_{shard_index}" + ext

    bar = tqdm(enumerate(source), desc="Saving dataset", disable=(not verbose))
    with h5py.File(path, "w") as f:
        for example_index, example in bar:
            example = (example,) if isinstance(example, Tensor) else example
            for i, tensor in enumerate(example):
                key = f"data_{i}"
                if key not in f.keys():
                    f.create_dataset(key, (shard_size, *tensor.shape))
                f[key][example_index, ...] = tensor

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


def _check_h5py():
    if h5py is None:
        raise ImportError(
            "HDF5 operations require h5py. "
            "Please install combustion with 'hdf5' extras using "
            "pip install combustion[hdf5]"
        )
    warnings.warn("hdf5 support is deprecated", DeprecationWarning)


__all__ = ["save_hdf5", "save_torch", "SerializeMixin", "HDF5Dataset", "TorchDataset", "TransformableDataset"]
