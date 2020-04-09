#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest


@pytest.fixture(scope="session")
def torch():
    return pytest.importorskip("torch", reason="test requires torch")


@pytest.fixture(scope="session")
def ignite():
    return pytest.importorskip("ignite", reason="test requires ignite")


@pytest.fixture(scope="session")
def cuda(torch):
    if not torch.cuda.is_available():
        pytest.skip("test requires cuda")


def pytest_report_header(config):
    try:
        import torch
        import ignite

        return "torch version: %s\nignite version: %s" % (torch.__version__, ignite.__version__,)
    except ImportError:
        return ""


@pytest.fixture(scope="session")
def np():
    return pytest.importorskip("numpy", reason="test requires numpy")


def pytest_addoption(parser):
    parser.addoption(
        "--tf", action="store_true", default=False, help="run tests requiring tensorflow",
    )
    parser.addoption(
        "--torch", action="store_true", default=False, help="run tests requiring pytorch",
    )
    parser.addoption(
        "--all", action="store_true", default=False, help="run all tests including torch/tf",
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
