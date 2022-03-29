#!/usr/bin/env python
# -*- coding: utf-8 -*-

import builtins
import os
import typing
from pathlib import Path
from shutil import copyfile

import pytest
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

from combustion.data import SerializeMixin, TorchDataset


def check_file_exists(filepath):
    __tracebackhide__ = True
    path = Path(filepath)
    contents = [x.name for x in path.parent.iterdir()]
    if not path.is_file():
        pytest.fail(f"file {path.name} not found in {path.parent}, contents {contents}")


class TestTorchSerialize:
    fmt = "torch"

    @pytest.fixture
    def data(self):
        return [(torch.rand(1, 10, 10), torch.randint(1, (1,))) for _ in range(10)]

    @pytest.fixture(params=["map", "iterable"])
    def dataset(self, request, data):
        if request.param == "iterable":

            class DatasetImpl(IterableDataset, SerializeMixin):  # type: ignore
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
    def input_file(self, tmp_path, data):
        for i, example in enumerate(data):
            path = os.path.join(tmp_path, f"example_{i}.pth")
            torch.save(example, path)
        return tmp_path

    @pytest.fixture
    def save_path(self, tmp_path):
        path = os.path.join(tmp_path, "foo")
        os.makedirs(path)
        return path

    @pytest.mark.parametrize(
        "prefix",
        [
            "example_",
            os.path.join("subdir", "example_"),
            pytest.param(
                lambda pos, example: os.path.join("subdir", f"example_{pos}.pth"), id="subdir/example_{pos}.pth"
            ),
            pytest.param(
                lambda pos, example: os.path.join(f"class_{example[1].item()}", f"example_{pos}.pth"),
                id="class_{label}/example_{pos}.pth",
            ),
        ],
    )
    def test_save(self, tmp_path, dataset, save_path, prefix):
        dataset.save(save_path, prefix=prefix)
        first_example = next(iter(dataset))
        if isinstance(prefix, str):
            target = os.path.join(save_path, f"{prefix}0.pth")
        else:
            target = os.path.join(save_path, f"{prefix(0, first_example)}.pth")
        check_file_exists(target)

    def test_create_directory_on_save(self, tmp_path, dataset, save_path):
        os.rmdir(save_path)
        dataset.save(save_path)
        target = os.path.join(save_path, "example_0.pth")
        check_file_exists(target)

    @pytest.mark.usefixtures("input_file")
    def test_load(self, tmp_path, dataset, data):
        path = tmp_path
        new_dataset: Dataset = dataset.__class__.load(path)
        assert isinstance(new_dataset, TorchDataset)
        for e1, e2 in zip(data, new_dataset):
            for t1, t2 in zip(e1, e2):
                assert torch.allclose(t1, t2)

        new_dataset = typing.cast(typing.Sized, new_dataset)  # type: ignore
        assert len([x for x in dataset]) == len([x for x in new_dataset])

    @pytest.mark.usefixtures("input_file")
    def test_load_pattern(self, tmp_path, dataset, data):
        src_files = os.listdir(tmp_path)
        for i, f in enumerate(src_files):
            p = os.path.join(tmp_path, f)
            new_p = os.path.join(tmp_path, f"copy_{i}.pth")
            copyfile(p, new_p)

        path = tmp_path
        new_dataset = dataset.__class__.load(path, pattern="example_*.pth")

        assert len(new_dataset) == len(data)
        for f in src_files:
            assert Path(tmp_path, f) in new_dataset.files

    @pytest.mark.parametrize("num_workers", [1, 4])
    def test_dataloader(self, tmp_path, dataset, num_workers):
        path = tmp_path
        new_dataset = dataset.__class__.load(path)
        dataloader = DataLoader(new_dataset, num_workers=num_workers, batch_size=1)

        for _ in range(10):
            for e1, e2 in zip(dataloader, new_dataset):
                for t1, t2 in zip(e1, e2):
                    assert torch.allclose(t1, t2)

    @pytest.mark.parametrize("num_workers", [1, 4])
    def test_save_threaded(self, dataset, save_path, num_workers):
        dataset.save(save_path, threads=num_workers)
        for i in range(len(dataset)):
            p = Path(save_path, f"example_{i}.pth")
            check_file_exists(p)

    @pytest.mark.usefixtures("input_file")
    @pytest.mark.parametrize("length", [1000, 2000])
    def test_length_override(self, tmp_path, dataset, length):
        path = tmp_path
        new_dataset = dataset.__class__.load(path, length_override=length)
        assert len(new_dataset) == length
        new_dataset[0]
        new_dataset[length - 1]

    def test_repr(self, dataset, input_file):
        path = input_file
        ds1 = dataset.__class__.load(path)
        repr(ds1)

        ds2 = dataset.__class__.load(path)
        repr(ds2)

    @pytest.mark.parametrize("verbose", [True, False])
    def test_verbose(self, dataset, mocker, verbose, save_path, capsys):
        mocker.spy(builtins, "print")
        dataset.save(save_path, verbose=verbose)
        captured = capsys.readouterr()

        if verbose:
            assert "Saving dataset" in captured.err
        else:
            assert not captured.err
        assert not captured.out

    def test_length(self, dataset, input_file, save_path):
        path = input_file
        dataset.save(save_path)
        ds = dataset.__class__.load(path)
        assert len(ds) == 10
