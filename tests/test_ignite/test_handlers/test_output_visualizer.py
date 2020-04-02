#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

import pytest

from combustion.ignite.handlers import OutputVisualizer


@pytest.fixture(params=["tuple", "single"])
def process_fn(torch, request, device):
    def f(state):
        if request.param == "tuple":
            return torch.rand(2, 1, 10, 10).to(device), torch.rand(2, 1, 10, 10).to(device)
        else:
            return torch.rand(2, 1, 10, 10).to(device)

    return f


@pytest.fixture
def visualize_fn(process_fn):
    plt = pytest.importorskip("matplotlib.pyplot")
    if isinstance(process_fn(None), tuple):

        def f(x, y, title):
            fig, axs = plt.subplots()
            fig.suptitle(title)
            plt.scatter(x, y)
            return fig

    else:

        def f(x, title):
            fig, axs = plt.subplots()
            fig.suptitle(title)
            plt.scatter(x, x)
            return fig

    return f


@pytest.fixture
def args(mock_args, tmpdir):
    mock_args.result_path = tmpdir
    mock_args.val_image_dir_fmt = "epoch_{epoch}"
    mock_args.val_image_file_fmt = "output_{iteration}.png"
    mock_args.val_image_title = "validation {epoch} {mse}"
    return mock_args


def test_saves_plot(tmpdir, engine, process_fn, visualize_fn):
    subdir = os.path.join(tmpdir, "epoch_{epoch}")
    filepath = os.path.join(tmpdir, "epoch_1", "output_1.png")
    vis = OutputVisualizer(subdir, "output_{iteration}.png", process_fn, visualize_fn, title="epoch {mse}")
    vis(engine)
    assert os.path.isfile(filepath)


def test_from_args(tmpdir, args, engine, process_fn, visualize_fn):
    vis = OutputVisualizer.from_args(args, process_fn, visualize_fn)
    assert vis.file_format == args.val_image_file_fmt
    assert vis.dir_format == os.path.join(args.result_path, args.val_image_dir_fmt)
    assert vis.title == args.val_image_title
    assert vis.process_fn == process_fn
    assert vis.visualize_fn == visualize_fn


def test_repr(tmpdir, engine, process_fn, visualize_fn):
    subdir = os.path.join(tmpdir, "epoch_{epoch}")
    file_fmt = "output_{iteration}.png"
    title = "epoch {mse}"
    vis = OutputVisualizer(subdir, file_fmt, process_fn, visualize_fn, title=title)
    out = repr(vis)
    assert subdir in out
    assert file_fmt in out
    assert title in out


def test_overwrite(tmpdir, engine, process_fn, visualize_fn):
    subdir = str(tmpdir)
    file_fmt = "output.png"
    title = "title"
    open(os.path.join(subdir, file_fmt), "a").close()
    vis = OutputVisualizer(subdir, file_fmt, process_fn, visualize_fn, title=title)
    with pytest.raises(FileExistsError):
        vis(engine)


def test_kwargs(tmpdir, engine, process_fn, visualize_fn):
    subdir = os.path.join(tmpdir, "epoch_{epoch}")
    filepath = os.path.join(tmpdir, "epoch_1", "output_1.png")
    vis = OutputVisualizer(subdir, "output_{iteration}.png", process_fn, visualize_fn, title="epoch {mse}", dpi=96)
    vis(engine)
    assert os.path.isfile(filepath)


def test_cuda(torch, tmpdir, engine, process_fn, visualize_fn):
    subdir = os.path.join(tmpdir, "epoch_{epoch}")
    filepath = os.path.join(tmpdir, "epoch_1", "output_1.png")
    vis = OutputVisualizer(subdir, "output_{iteration}.png", process_fn, visualize_fn, title="epoch {mse}", dpi=96)
    vis(engine)
    assert os.path.isfile(filepath)
