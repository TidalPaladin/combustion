#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

import pytest

from combustion.data.dataset import MatlabDataset
from combustion.data.window import DenseWindow, SparseWindow


@pytest.fixture(autouse=True)
def args(mock_args, tmpdir):
    mock_args.data_path = tmpdir
    mock_args.matlab_data_key = "data_key"
    mock_args.matlab_label_key = "label_key"
    mock_args.dense_window = None
    mock_args.sparse_window = None
    return mock_args


@pytest.fixture
def data(args, torch, request):
    data = torch.ones(12, 8, 4).detach().numpy()
    labels = torch.zeros(12, 8, 4).detach().numpy()
    return {args.matlab_data_key: data, args.matlab_label_key: labels}


@pytest.fixture
def matlab_file(request, args, data, matlab_saver):
    filepath = os.path.join(args.data_path, "test_file.mat")
    matlab_saver(filepath, data)
    return filepath


@pytest.fixture(
    params=[pytest.param(None, id="no_xform"), pytest.param(lambda frames, labels: (frames, labels), id="xform"),]
)
def transform(request):
    return request.param


@pytest.fixture(
    params=[
        pytest.param(None, id="no_window"),
        pytest.param(DenseWindow(2, 2), id="Dense(2,2)"),
        pytest.param(SparseWindow(2, 2), id="Sparse(2,2)"),
    ]
)
def window(request):
    return request.param


def test_constructor(args, data, matlab_saver, transform, window):
    filepath = os.path.join(args.data_path, "test_file.mat")
    matlab_saver(filepath, data)
    MatlabDataset(filepath, args.matlab_data_key, args.matlab_label_key, transform=transform, window=window)


def test_length(matlab_file, args, data, window):
    dataset = MatlabDataset(matlab_file, args.matlab_data_key, args.matlab_label_key, window=window)
    if window is None:
        expected = data[args.matlab_data_key].shape[0]
    else:
        expected = data[args.matlab_data_key].shape[0] - 4
    assert len(dataset) == expected


def test_getitem(matlab_file, args, data, torch, transform, window):
    dataset = MatlabDataset(
        matlab_file, args.matlab_data_key, args.matlab_label_key, transform=transform, window=window
    )

    if window is None:
        expected = (data[args.matlab_data_key][0], data[args.matlab_label_key][0])
    elif isinstance(window, SparseWindow):
        expected = (data[args.matlab_data_key][0:3], data[args.matlab_label_key][1])
    else:
        expected = (data[args.matlab_data_key][0:5], data[args.matlab_label_key][2])

    example = dataset[0]
    expected = torch.as_tensor(expected[0]).unsqueeze(0), torch.as_tensor(expected[1]).unsqueeze(0)

    if window is None:
        expected = expected[0].unsqueeze(0), expected[1].unsqueeze(0)

    assert example[0].shape == expected[0].shape, "data mismatch"
    assert example[0].shape == expected[0].shape, "data mismatch"
    assert torch.allclose(example[0].rename(None), expected[0])
    assert torch.allclose(example[1].rename(None), expected[1])


def test_iter(args, matlab_file, data, torch, transform, window):
    dataset = MatlabDataset(
        matlab_file, args.matlab_data_key, args.matlab_label_key, transform=transform, window=window
    )
    output = list(iter(dataset))
    assert len(output) == len(dataset)
    for frames, label in output:
        assert frames.requires_grad
