#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

import pytest
import torch
from torch import Tensor

from combustion.testing import cuda_or_skip as cuda_or_skip_mark
from combustion.testing.utils import cuda_available
from typing import Any, Optional, Sequence, Tuple, Union


@pytest.fixture(
    params=[
        pytest.param(False, id="no_cuda"),
        pytest.param(True, marks=cuda_or_skip_mark, id="cuda"),
    ]
)
def cuda(request):
    return request.param


@pytest.fixture(scope="session")
def cuda_or_skip():
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


@pytest.fixture
def padded_coords_factory():
    def func(
        traces: int = 3,
        trace_len: int = 10,
        batch_size: Optional[int] = None,
        seed: int = 42,
        pad_val: Any = 0,
        cuda: bool = False,
        requires_grad: bool = False,
        lower_bound: Union[float, Sequence[float]] = 0.0,
        upper_bound: Union[float, Sequence[float]] = 1.0,
        coord_dims: int = 2,
    ) -> Tuple[Tensor, Tensor]:
        MIN_TRACE_LEN = 3
        assert trace_len >= MIN_TRACE_LEN
        shape = (traces, trace_len, coord_dims)
        torch.random.manual_seed(seed)
        data = torch.rand(*shape)

        low = data.new_tensor(lower_bound).broadcast_to(data.shape)
        high = data.new_tensor(upper_bound).broadcast_to(data.shape)
        data = (data + low) * (high - low)
        data[..., 0, :] = data.new_tensor(lower_bound)
        data[..., 1, :] = data.new_tensor(upper_bound)

        coord_id = torch.arange(trace_len).view(1, -1, 1).expand(traces, -1, coord_dims)
        limit = torch.randint(MIN_TRACE_LEN, trace_len + 1, (traces, 1, 1))
        padding = (coord_id > limit).any(dim=-1)
        data[padding] = pad_val

        if batch_size is not None:
            data = data.unsqueeze_(0).expand(batch_size, -1, -1, -1)
            padding = padding.unsqueeze_(0).expand(batch_size, -1, -1)

        if cuda:
            data = data.cuda()
            padding = padding.cuda()

        if requires_grad:
            data.requires_grad = True

        return data, padding

    return func
