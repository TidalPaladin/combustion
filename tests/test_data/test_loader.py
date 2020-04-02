#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob
import os

import pytest

from combustion.data.loader import load_from_args, load_matlab


class TestMatlabLoader:
    data_key = "data"
    label_key = "label"

    @pytest.fixture
    def filepath(self, tmpdir, data):
        sio = pytest.importorskip("scipy.io")
        filepath = os.path.join(tmpdir, "test_file.mat")
        sio.savemat(filepath, data)
        return filepath

    @pytest.fixture(
        params=[
            pytest.param((8, 5, 5), id="8x5x5"),
            pytest.param((10, 4, 5), id="10x4x5"),
            pytest.param((2, 2), id="2x2"),
        ]
    )
    def data(self, torch, request):
        data = torch.ones(request.param).detach().numpy()
        labels = torch.zeros(request.param).detach().numpy()
        return {self.data_key: data, self.label_key: labels}

    def test_load_data(self, filepath, data, torch):
        loaded_data, loaded_labels = load_matlab(filepath, data_key=self.data_key, label_key=self.label_key)
        assert isinstance(loaded_data, torch.Tensor)
        assert isinstance(loaded_labels, torch.Tensor)
        assert (data[self.data_key] == loaded_data.detach().numpy()).all()
        assert (data[self.label_key] == loaded_labels.detach().numpy()).all()

    def test_validates_data_label_shape_match(self, torch, tmpdir):
        # setup invalid file with dim mismatch
        sio = pytest.importorskip("scipy.io")
        filepath = os.path.join(tmpdir, "test_file.mat")
        data = torch.ones((8, 5, 5)).detach().numpy()
        labels = torch.zeros((9, 5, 5)).detach().numpy()
        matlab_data = {self.data_key: data, self.label_key: labels}
        sio.savemat(filepath, matlab_data)

        with pytest.raises(AssertionError):
            load_matlab(filepath, data_key=self.data_key, label_key=self.label_key)

    def test_transpose(self, filepath, data, np, torch):
        data, labels = data[self.data_key], data[self.label_key]
        if len(data) > 2:
            transpose = (1, 2, 0)
        else:
            transpose = (1, 0)
        out_data = torch.as_tensor(data.transpose(transpose))
        out_labels = torch.as_tensor(labels.transpose(transpose))
        actual_data, actual_labels = load_matlab(
            filepath, data_key=self.data_key, label_key=self.label_key, transpose=transpose
        )
        assert (actual_data == out_data).all()
        assert (actual_labels == out_labels).all()


class TestLoadFromArgs:
    @pytest.fixture(autouse=True)
    def args(self, mock_args, tmpdir):
        mock_args.data_path = tmpdir
        mock_args.matlab_data_key = "data_key"
        mock_args.matlab_label_key = "label_key"
        mock_args.dense_window = 0
        mock_args.sparse_window = 0
        return mock_args

    @pytest.fixture(
        params=[
            pytest.param((8, 5, 5), id="8x5x5"),
            pytest.param((10, 4, 5), id="10x4x5"),
            pytest.param((2, 2), id="2x2"),
        ]
    )
    def data(self, args, torch, request):
        data = torch.ones(request.param).detach().numpy()
        labels = torch.zeros(request.param).detach().numpy()
        return {args.matlab_data_key: data, args.matlab_label_key: labels}

    @pytest.fixture(
        params=[pytest.param(1, id="num_matlab=1"), pytest.param(2, id="num_matlab=2"),]
    )
    def matlab_files(self, request, args, data, matlab_saver):
        sio = pytest.importorskip("scipy.io")
        for i in range(request.param):
            filepath = os.path.join(args.data_path, "test_file_%s.mat" % i)
            matlab_saver(filepath, data)
        return os.listdir(args.data_path)

    @pytest.fixture(
        params=[pytest.param(1, id="num_pth=1"), pytest.param(2, id="num_pth=2"),]
    )
    def pth_files(self, torch, request, args, data):
        for i in range(request.param):
            filepath = os.path.join(args.data_path, "test_file_%s.pth" % i)
            torch.save(data, filepath)
        return glob.glob(os.path.join(args.data_path, "*.pth"))

    def test_file_not_found_exception_if_no_match(self, args):
        with pytest.raises(FileNotFoundError):
            load_from_args(args)

    def test_load_matlab(self, matlab_files, args, torch, data):
        result = load_from_args(args)
        assert isinstance(result, torch.utils.data.Dataset)
        assert len(result) == len(matlab_files) * len(data[args.matlab_data_key])
