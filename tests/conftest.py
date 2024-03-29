#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

import pytest

from combustion.testing import cuda_or_skip as cuda_or_skip_mark
from combustion.testing.utils import cuda_available


@pytest.fixture(scope="session")
def torch():
    return pytest.importorskip("torch", reason="test requires torch")


@pytest.fixture(scope="session")
def ignite():
    return pytest.importorskip("ignite", reason="test requires ignite")


@pytest.fixture(
    params=[
        pytest.param(False, id="no_cuda"),
        pytest.param(True, marks=cuda_or_skip_mark, id="cuda"),
    ]
)
def cuda(torch, request):
    return request.param


@pytest.fixture(scope="session")
def cuda_or_skip(torch):
    if not cuda_available():
        pytest.skip("test requires cuda")


def pytest_report_header(config):
    import hydra
    import pytorch_lightning
    import torch

    s = "Version Information: \n"
    s += f"torch: {torch.__version__}\n"
    s += f"pytorch_lightning: {pytorch_lightning.__version__}\n"
    s += f"hydra: {hydra.__version__}\n"

    try:
        import kornia
        import torchvision

        s += f"torchvision: {torchvision.__version__}\n"
        s += f"kornia: {kornia.__version__}\n"
    except ImportError:
        pass

    return s


@pytest.fixture(scope="session")
def np():
    return pytest.importorskip("numpy", reason="test requires numpy")


def pytest_addoption(parser):
    parser.addoption(
        "--tf",
        action="store_true",
        default=False,
        help="run tests requiring tensorflow",
    )
    parser.addoption(
        "--torch",
        action="store_true",
        default=False,
        help="run tests requiring pytorch",
    )
    parser.addoption(
        "--all",
        action="store_true",
        default=False,
        help="run all tests including torch/tf",
    )


def pytest_runtest_setup(item):
    requires_tf = bool(next(item.iter_markers(name="requires_tf"), False))
    requires_torch = bool(next(item.iter_markers(name="requires_torch"), False))
    requires_neither = not (requires_tf or requires_torch)

    testing_tf = item.config.getoption("--tf")
    testing_torch = item.config.getoption("--torch")
    testing_neither = not (testing_tf or testing_torch)

    if item.config.getoption("--all"):
        return
    if requires_tf and not testing_tf:
        pytest.skip("test requires tensorflow")
    if requires_torch and not testing_torch:
        pytest.skip("test requires pytorch")
    if requires_neither and not testing_neither:
        pytest.skip("test foo")


@pytest.fixture
def mock_args(mocker):
    m = mocker.MagicMock(name="args")
    m.log_format = "[%(asctime)s %(levelname).1s] - %(message)s"
    mocker.patch("combustion.args.parse_args", return_value=m)
    return m


@pytest.fixture(scope="session")
def matlab_saver():
    h5py = pytest.importorskip("h5py")

    def func(path, to_save):
        f = h5py.File(path, "w")
        for k, v in to_save.items():
            f.create_dataset(k, data=v)
            f.flush()
        f.close()

    return func


@pytest.fixture(autouse=True)
def chdir_to_tmp_path(tmp_path):
    os.chdir(tmp_path)
