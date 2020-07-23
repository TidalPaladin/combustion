#!/usr/bin/env python
# -*- coding: utf-8 -*-

import builtins
import math
import os
from pathlib import Path
from shutil import copyfile

import pytest
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

# from combustion.data import AbstractDataset
from combustion.data import HDF5Dataset, SerializeMixin, TorchDataset


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

    fmt = "hdf5"

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
    def save_path(self, tmp_path):
        return os.path.join(tmp_path, "foo.h5")

    @pytest.fixture
    def input_file(self, h5py, torch, tmp_path, data, save_path):
        path = save_path
        f = h5py.File(path, "w")
        f.create_dataset("data_0", (len(data), *data[0][0].shape))
        f.create_dataset("data_1", (len(data), *data[0][1].shape))
        for i, ex in enumerate(data):
            f["data_0"][i, ...] = ex[0]
            f["data_1"][i, ...] = ex[1]
        return path

    def test_save(self, h5py, tmp_path, dataset, save_path):
        dataset.save(save_path, fmt=self.fmt)
        check_file_exists(save_path)

    def test_load(self, h5py, torch, tmp_path, dataset, input_file, data):
        os.path.join(tmp_path, "foo.h5")
        new_dataset = dataset.__class__.load(input_file)
        assert isinstance(new_dataset, HDF5Dataset)
        for e1, e2 in zip(data, new_dataset):
            for t1, t2 in zip(e1, e2):
                assert torch.allclose(t1, t2)

    def test_preserves_attributes(self, h5py, tmp_path, dataset, save_path):
        shard = 0
        setattr(dataset, "shard", shard)
        dataset.save(save_path, fmt=self.fmt)
        check_file_exists(save_path)
        new_dataset = dataset.__class__.load(save_path)
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
    def test_save_shards(self, h5py, tmp_path, dataset, num_shards, shard_size, save_path):
        pattern = os.path.join(tmp_path, "foo_{shard}.h5")
        path = os.path.join(tmp_path, "foo.h5")
        dataset.save(path, fmt=self.fmt, num_shards=num_shards, shard_size=shard_size)
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

    def test_set_shard_metadata(self, h5py, tmp_path, dataset, save_path):
        pattern = os.path.join(tmp_path, "foo_{shard}.h5")
        path = os.path.join(tmp_path, "foo.h5")
        num_shards = 2
        dataset.save(path, fmt=self.fmt, num_shards=num_shards)

        for shard in range(1, num_shards + 1):
            path = pattern.format(shard=shard, num_shards=num_shards)
            with h5py.File(path, "r") as f:
                assert "shard_index" in f.attrs.keys()

    @pytest.mark.parametrize("transform", [lambda x: torch.ones(2, 2), None])
    @pytest.mark.parametrize("target_transform", [lambda x: torch.zeros(2, 2), None])
    def test_apply_transforms_to_loaded_data(self, h5py, tmp_path, dataset, transform, target_transform, input_file):
        new_dataset = dataset.__class__.load(input_file, transform=transform, target_transform=target_transform)
        example = next(iter(new_dataset))

        if transform is not None:
            assert torch.allclose(example[0], transform(None))
        if target_transform is not None:
            assert torch.allclose(example[1], target_transform(None))

    def test_repr(self, h5py, torch, tmp_path, dataset, input_file, data):
        path = input_file
        ds1 = dataset.__class__.load(path)
        repr(ds1)

        def xform(x):
            return 10

        ds2 = dataset.__class__.load(path, transform=xform, target_transform=xform)
        print(repr(ds2))

    @pytest.mark.parametrize("verbose", [True, False])
    def test_verbose(self, h5py, tmp_path, dataset, mocker, verbose, save_path):
        spy = mocker.spy(builtins, "print")
        dataset.save(save_path, fmt=self.fmt, verbose=verbose)

        if verbose:
            spy.assert_called()
        else:
            spy.assert_not_called()

    def test_multiple_loops(self, h5py, tmp_path, dataset, save_path):
        os.path.join(tmp_path, "foo_{shard}.h5")
        os.path.join(tmp_path, "foo.h5")
        dataset.save(save_path, fmt=self.fmt)

        for example in dataset:
            pass

        for example in dataset:
            pass

    def test_length(self, h5py, tmp_path, dataset, input_file, save_path):
        path = input_file
        dataset.save(save_path, fmt=self.fmt)
        ds = dataset.__class__.load(path)
        assert len(ds) == 10

    @pytest.mark.ci_skip
    @pytest.mark.parametrize(
        "num_workers", [pytest.param(1), pytest.param(4, marks=pytest.mark.xfail(reason="parallel hdf5")),],
    )
    def test_dataloader(self, h5py, torch, tmp_path, dataset, input_file, data, num_workers):
        path = input_file
        new_dataset = dataset.__class__.load(path)
        dataloader = DataLoader(new_dataset, batch_size=1)

        for i in range(100):
            for e1, e2 in zip(dataloader, new_dataset):
                for t1, t2 in zip(e1, e2):
                    assert torch.allclose(t1, t2)


class TestTorchSerialize(TestSerialize):
    fmt = "torch"

    @pytest.fixture
    def input_file(self, torch, tmp_path, data):
        for i, example in enumerate(data):
            path = os.path.join(tmp_path, f"example_{i}.pth")
            torch.save(example, path)
        return tmp_path

    @pytest.fixture
    def save_path(self, tmp_path):
        path = os.path.join(tmp_path, "foo")
        os.makedirs(path)
        return path

    def test_save(self, h5py, tmp_path, dataset, save_path):
        dataset.save(save_path, fmt=self.fmt)
        target = os.path.join(save_path, "example_0.pth")
        check_file_exists(target)

    def test_create_directory_on_save(self, h5py, tmp_path, dataset, save_path):
        os.rmdir(save_path)
        dataset.save(save_path, fmt=self.fmt)
        target = os.path.join(save_path, "example_0.pth")
        check_file_exists(target)

    def test_load(self, torch, tmp_path, dataset, input_file, data):
        path = tmp_path
        new_dataset = dataset.__class__.load(path)
        assert isinstance(new_dataset, TorchDataset)
        for e1, e2 in zip(data, new_dataset):
            for t1, t2 in zip(e1, e2):
                assert torch.allclose(t1, t2)

    def test_load_pattern(self, torch, tmp_path, dataset, input_file, data):
        src_files = os.listdir(tmp_path)
        for i, f in enumerate(src_files):
            p = os.path.join(tmp_path, f)
            new_p = os.path.join(tmp_path, f"copy_{i}.pth")
            copyfile(p, new_p)

        path = tmp_path
        new_dataset = dataset.__class__.load(path, pattern="example_*.pth")

        assert len(new_dataset) == len(data)
        for f in src_files:
            assert os.path.join(tmp_path, f) in new_dataset.files

    @pytest.mark.parametrize("num_workers", [1, 4])
    def test_dataloader(self, h5py, torch, tmp_path, dataset, input_file, data, num_workers):
        path = tmp_path
        new_dataset = dataset.__class__.load(path)
        dataloader = DataLoader(new_dataset, num_workers=num_workers, batch_size=1)

        for i in range(10):
            for e1, e2 in zip(dataloader, new_dataset):
                for t1, t2 in zip(e1, e2):
                    assert torch.allclose(t1, t2)

    def test_preserves_attributes(self):
        pass

    def test_save_shards(self):
        pass

    def test_set_shard_metadata(self):
        pass
