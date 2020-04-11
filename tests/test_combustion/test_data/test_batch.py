#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest

from combustion.data import Batch


class MyBatch(Batch):
    pass


@pytest.fixture
def batch_size():
    return 10


@pytest.fixture
def frames(batch_size, torch):
    return [torch.ones(3, 9, 9) for x in range(batch_size)]


@pytest.fixture
def labels(batch_size, torch):
    return [torch.zeros(3, 9, 9) for x in range(batch_size)]


@pytest.fixture
def data(frames, labels):
    return [(f, l) for f, l in zip(frames, labels)]


def test_attributes(data, frames, labels, batch_size, torch):
    batch = MyBatch(frames=frames, labels=labels)
    assert (batch.labels == torch.stack(labels, 0)).all()
    assert len(batch.labels) == batch_size
    assert (batch.frames == torch.stack(frames, 0)).all()
    assert len(batch.frames) == batch_size


def test_tuple_expansion(data, frames, labels, batch_size, torch):
    batch = MyBatch(frames=frames, labels=labels)
    frames, labels = batch
    assert frames is batch.frames
    assert labels is batch.labels


def test_tensor_slice(data, batch_size, torch):
    f = torch.rand(10, 10)
    l = torch.rand(10, 10)
    batch = MyBatch(frames=f, labels=l)
    frames, labels = batch[0]
    assert (frames == f[0]).all()
    assert (labels == l[0]).all()


def test_len(data, frames, labels, batch_size, torch):
    batch = MyBatch(frames=frames, labels=labels)
    assert len(batch) == len(frames)


def test_repr(data, frames, labels, batch_size, torch):
    batch = MyBatch(frames=frames, labels=labels)
    rep = str(batch)
    assert "frames" in rep
    assert "labels" in rep


def test_collate(data, frames, labels, batch_size, torch):
    with pytest.raises(NotImplementedError):
        MyBatch.collate_fn(data)


def test_getattr_func(frames, labels, batch_size, torch, mocker):
    batch = MyBatch(frames=frames, labels=labels)
    batch.byte()
    assert isinstance(batch.frames, torch.ByteTensor)
    assert isinstance(batch.labels, torch.ByteTensor)


def test_apply(frames, labels, batch_size, torch, mocker):
    batch = MyBatch(frames=frames, labels=labels)
    batch.apply(lambda x: x.byte())
    assert isinstance(batch.frames, torch.ByteTensor)
    assert isinstance(batch.labels, torch.ByteTensor)


def test_exception_no_kwargs_given(data, frames, labels, batch_size, torch):
    with pytest.raises(ValueError):
        MyBatch()
