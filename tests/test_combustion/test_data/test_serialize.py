#!/usr/bin/env python
# -*- coding: utf-8 -*-

import builtins
import math
import os
from pathlib import Path

import pytest
import torch
from torch.utils.data import Dataset, IterableDataset

# from combustion.data import AbstractDataset
from combustion.data import HDF5Dataset, SerializeMixin


def check_file_exists(filepath):
    __tracebackhide__ = True
    path = Path(filepath)
    contents = [x.name for x in path.parent.iterdir()]
    if not path.is_file():
        pytest.fail(f"file {path.name} not found in {path.parent}, contents {contents}")


@pytest.fixture
def h5py():
    return pytest.importorskip("h5py", reason="test requires h5py")


class TestSerialize:
    @pytest.fixture
    def data(self, torch):
        return [(torch.rand(1, 10, 10), torch.rand(1, 5)) for x in range(10)]

    @pytest.fixture(params=["map", "iterable"])
    def dataset(self, request, torch, data):
        if request.param == "iterable":

            class DatasetImpl(IterableDataset, SerializeMixin):
                def __iter__(self):
                    return iter(data)

                def __len__(self):
                    return len(data)

        else:

            class DatasetImpl(Dataset, SerializeMixin):
                def __getitem__(self, index):
                    return data[index]

                def __len__(self):
                    return len(data)

        return DatasetImpl()

    @pytest.fixture
    def input_file(self, h5py, torch, tmp_path, data):
        path = os.path.join(tmp_path, "foo.pth")
        f = h5py.File(path, "w")
        f.create_dataset("data_0", (len(data), *data[0][0].shape))
        f.create_dataset("data_1", (len(data), *data[0][1].shape))
        for i, ex in enumerate(data):
            f["data_0"][i, ...] = ex[0]
            f["data_1"][i, ...] = ex[1]
        return path

    def test_save(self, h5py, tmp_path, dataset):
        path = os.path.join(tmp_path, "foo.pth")
        dataset.save(path)
        check_file_exists(path)

    def test_load(self, h5py, torch, tmp_path, dataset, input_file, data):
        path = os.path.join(tmp_path, "foo.pth")
        new_dataset = dataset.__class__.load(path)
        assert isinstance(new_dataset, HDF5Dataset)
        for e1, e2 in zip(data, new_dataset):
            for t1, t2 in zip(e1, e2):
                assert torch.allclose(t1, t2)

    def test_preserves_attributes(self, h5py, tmp_path, dataset):
        path = os.path.join(tmp_path, "foo.pth")
        shard = 0
        setattr(dataset, "shard", shard)
        dataset.save(path)
        check_file_exists(path)
        new_dataset = dataset.__class__.load(path)
        assert hasattr(new_dataset, "shard")
        assert new_dataset.shard == shard

    @pytest.mark.parametrize(
        "num_shards,shard_size",
        [
            pytest.param(2, None),
            pytest.param(None, 5),
            pytest.param(2, 2, marks=pytest.mark.xfail(raises=ValueError)),
            pytest.param(0, None, marks=pytest.mark.xfail(raises=ValueError)),
            pytest.param(None, 0, marks=pytest.mark.xfail(raises=ValueError)),
            pytest.param(None, None),
        ],
    )
    def test_save_shards(self, h5py, tmp_path, dataset, num_shards, shard_size):
        pattern = os.path.join(tmp_path, "foo_{shard}.pth")
        path = os.path.join(tmp_path, "foo.pth")
        dataset.save(path, num_shards=num_shards, shard_size=shard_size)
        if num_shards is None and shard_size is not None:
            num_shards = math.ceil(len(dataset) // shard_size)
        elif shard_size is None and num_shards is not None:
            shard_size = math.ceil(len(dataset) // num_shards)
        else:
            num_shards = 1

        check_file_exists(path)  # check parent virtual dataset created
        if num_shards > 1:
            for shard in range(1, num_shards + 1):
                check_file_exists(pattern.format(shard=shard, num_shards=num_shards))

    def test_set_shard_metadata(self, h5py, tmp_path, dataset):
        pattern = os.path.join(tmp_path, "foo_{shard}.pth")
        path = os.path.join(tmp_path, "foo.pth")
        num_shards = 2
        dataset.save(path, num_shards=num_shards)

        for shard in range(1, num_shards + 1):
            path = pattern.format(shard=shard, num_shards=num_shards)
            with h5py.File(path, "r") as f:
                assert "shard_index" in f.attrs.keys()

    @pytest.mark.parametrize("transform", [lambda x: torch.ones(2, 2), None])
    @pytest.mark.parametrize("target_transform", [lambda x: torch.zeros(2, 2), None])
    def test_apply_transforms_to_loaded_data(self, h5py, tmp_path, dataset, transform, target_transform):
        path = os.path.join(tmp_path, "foo.pth")
        dataset.save(path)
        new_dataset = dataset.__class__.load(path, transform=transform, target_transform=target_transform)
        example = next(iter(new_dataset))

        if transform is not None:
            assert torch.allclose(example[0], transform(None))
        if target_transform is not None:
            assert torch.allclose(example[1], target_transform(None))

    def test_repr(self, h5py, torch, tmp_path, dataset, input_file, data):
        path = os.path.join(tmp_path, "foo.pth")
        ds1 = dataset.__class__.load(path)
        repr(ds1)

        def xform(x):
            return 10

        ds2 = dataset.__class__.load(path, transform=xform, target_transform=xform)
        print(repr(ds2))

    @pytest.mark.parametrize("verbose", [True, False])
    def test_verbose(self, h5py, tmp_path, dataset, mocker, verbose):
        path = os.path.join(tmp_path, "foo.pth")
        spy = mocker.spy(builtins, "print")
        dataset.save(path, verbose=verbose)

        if verbose:
            spy.assert_called()
        else:
            spy.assert_not_called()

    def test_multiple_loops(self, h5py, tmp_path, dataset):
        os.path.join(tmp_path, "foo_{shard}.pth")
        path = os.path.join(tmp_path, "foo.pth")
        dataset.save(path)

        for example in dataset:
            pass

        for example in dataset:
            pass

    def test_length(self, h5py, tmp_path, dataset):
        os.path.join(tmp_path, "foo_{shard}.pth")
        path = os.path.join(tmp_path, "foo.pth")
        dataset.save(path)
        ds = dataset.__class__.load(path)
        assert len(ds) == 10
