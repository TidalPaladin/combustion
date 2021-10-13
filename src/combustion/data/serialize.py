#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import typing
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sized, Tuple, Union

import torch
from torch import Tensor
from tqdm import tqdm


def save_torch(
    dataset: Iterable[Any],
    path: Union[Path, str],
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

    total: Optional[int]
    if hasattr(dataset, "__len__"):
        total = len(typing.cast(Sized, dataset))
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

    dataset = typing.cast(Iterable, dataset)
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
        if not isinstance(self, Iterable) and not hasattr(self, "__getitem__"):
            raise TypeError(f"Expected self to be an iterable, found {type(self)}")
        self = typing.cast(Iterable, self)
        save_torch(self, path=path, prefix=prefix, verbose=verbose, threads=threads)

    @staticmethod
    def load(
        path: str,
        **kwargs,
    ) -> TorchDataset:
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
        return TorchDataset(path, **kwargs)


class TorchDataset(SerializeMixin):
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
        path: Union[str, Path],
        length_override: Optional[int] = None,
        pattern: str = "*.pth",
        transform: Optional[Callable] = None,
    ):
        super().__init__()
        self.path = Path(path)
        self.pattern = pattern
        self.files = sorted(list(Path(self.path).glob(self.pattern)))
        self.transform = transform
        if length_override is not None:
            length_override = int(length_override)
            if length_override <= 0:
                raise ValueError(f"Expected length_override > 0, found {length_override}")
            self.length_override = length_override
        else:
            self.length_override = None

    def __repr__(self):
        rep = f"TorchDataset({self.path}"
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
        return example

    def __len__(self):
        if self.length_override is not None:
            return self.length_override
        else:
            return len(self.files)

    def apply_transform(self, example) -> Any:
        if self.transform is None:
            return example
        return self.transform(*example)


__all__ = ["save_torch", "SerializeMixin", "TorchDataset"]
