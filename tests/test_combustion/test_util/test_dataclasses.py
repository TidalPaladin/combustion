#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dataclasses import dataclass, field
from typing import Optional, Sequence

import pytest
import torch
from torch import Tensor
from torch.testing import assert_allclose

from combustion.testing import cuda_or_skip
from combustion.util.compute import slice_along_dim
from combustion.util.dataclasses import BatchMixin, TensorDataclass, pad, padded_stack, trim_padding


@pytest.mark.parametrize(
    "dim,slice_val,slice_fn",
    [
        pytest.param(0, slice(0, 5), lambda x, s: x[s, ...]),
        pytest.param(1, slice(0, 5), lambda x, s: x[:, s, ...]),
        pytest.param(2, slice(0, 5), lambda x, s: x[:, :, s, ...]),
        pytest.param(-1, slice(0, 5), lambda x, s: x[..., s]),
        pytest.param(0, slice(5), lambda x, s: x[s, ...]),
        pytest.param(0, slice(5, None, 2), lambda x, s: x[s, ...]),
    ],
)
def test_slice_along_dim(slice_val, dim, slice_fn):
    torch.random.manual_seed(42)
    x = torch.rand(*[10] * 4)
    expected = slice_fn(x, slice_val)
    actual = slice_along_dim(x, dim, slice_val.start, slice_val.stop, slice_val.step)
    assert torch.allclose(expected, actual)


@dataclass(repr=False)
class DummyDataclass(TensorDataclass, BatchMixin):
    data: Tensor
    label: Optional[Tensor] = None
    sparse_field: Optional[Tensor] = None
    ragged_field: Optional[Tensor] = None
    sequence: Sequence[int] = field(default_factory=lambda: [1])
    other_field: str = "foobar"

    __slice_fields__ = ["data", "label", "sequence", "sparse_field", "ragged_field"]

    def __post_init__(self):
        self._validate_slice_fields()
        assert self.sparse_field is None or self.sparse_field.is_sparse

    @property
    def is_batched(self) -> bool:
        return self.data.ndim == 4

    @classmethod
    def from_unbatched(cls, examples) -> "DummyDataclass":
        ...


@pytest.fixture
def target_factory(padded_coords_factory):
    def wrapper(
        batch_size: Optional[int] = None,
        has_label: bool = False,
        cuda: bool = False,
        requires_grad: bool = False,
        **kwargs,
    ):
        coords, _ = padded_coords_factory(batch_size=batch_size, cuda=cuda, **kwargs)
        sparse = coords.to_sparse(coords.ndim)

        device = "cuda:0" if cuda else "cpu"
        if batch_size:
            data = torch.rand(batch_size, 3, 10, 10, device=device, requires_grad=requires_grad)
            label = torch.rand(batch_size, 1, device=device) if has_label else None
            sequence = [1] * batch_size
        else:
            data = torch.rand(3, 10, 10, device=device, requires_grad=requires_grad)
            label = torch.rand(1, device=device) if has_label else None
            sequence = [1]
        return DummyDataclass(data, label, sparse, coords, sequence)

    return wrapper


class TestBatchMixin:
    @pytest.mark.parametrize("has_label", [False, True])
    def test_repr(self, target_factory, has_label):
        dc = target_factory(has_label=has_label)
        s = str(dc)
        assert isinstance(s, str)

    @pytest.mark.parametrize("has_label", [False, True])
    @pytest.mark.parametrize("batch_size", [None, 2])
    @pytest.mark.parametrize("traces", [3, 5])
    def test_len(self, target_factory, has_label, batch_size, traces):
        dc = target_factory(batch_size, has_label, traces=traces)
        assert len(dc) == batch_size if batch_size else traces

    @pytest.mark.parametrize("has_label", [False, True])
    def test_stack(self, target_factory, has_label):
        dc1 = target_factory(has_label=has_label)
        dc2 = target_factory(has_label=has_label)
        out: BatchMixin = torch.stack([dc1, dc2])  # type: ignore
        assert out.is_batched
        assert len(out) == 2

    @pytest.mark.parametrize("has_label", [False, True])
    def test_collate(self, target_factory, has_label):
        dc1 = target_factory(has_label=has_label, traces=5)
        dc2 = target_factory(has_label=has_label, traces=10)
        out: BatchMixin = DummyDataclass.collate([dc1, dc2])
        assert out.is_batched
        assert len(out) == 2
        out1 = out[0]
        out2 = out[1]

        assert out1.sequence == dc1.sequence
        assert_allclose(out1.data, dc1.data)
        assert_allclose(out1.ragged_field, dc1.ragged_field)
        assert_allclose(out1.sparse_field, dc1.sparse_field)

        assert out2.sequence == dc2.sequence
        assert_allclose(out2.data, dc2.data)
        assert_allclose(out2.ragged_field, dc2.ragged_field)
        assert_allclose(out2.sparse_field, dc2.sparse_field)

    @pytest.mark.parametrize(
        "ndim,lengths,dim",
        [
            pytest.param(3, [3, 4, 5, 3], 1),
            pytest.param(3, [3, 4, 5, 3], 0),
        ],
    )
    def test_padded_stack(self, ndim, lengths, dim):
        torch.random.manual_seed(42)
        max_len = max(lengths)
        longest_index = lengths.index(max_len)

        sizes = [[10] * ndim for l in lengths]
        for s, l in zip(sizes, lengths):
            s[dim] = l
        tensors = [torch.rand(*size) for size in sizes]
        max_tensor = tensors[longest_index]

        result = padded_stack(tensors, pad_value=-1, dim=0)
        assert result.shape

        assert result.ndim == ndim + 1
        assert result.shape[dim + 1] == max_len
        assert torch.allclose(result[longest_index], max_tensor)

    def test_pad(self):
        torch.random.manual_seed(42)
        x1 = torch.rand(1, 32, 8)
        x2 = torch.rand(3, 18, 19)

        expected = [3, 32, 19]
        p1 = pad(x1, expected)
        p2 = pad(x2, expected)
        expected = tuple(expected)

        assert p1.shape == expected
        assert p2.shape == expected
        assert p1[0, 0, 0] != 0
        assert p1[0, 0, -1] == 0
        assert p2[0, 0, 0] != 0
        assert p2[0, -1, 0] == 0
        assert torch.allclose(p1.sum(), x1.sum())
        assert torch.allclose(p2.sum(), x2.sum())

    def test_pad_zero_dim(self):
        torch.random.manual_seed(42)
        x1 = torch.empty(0, 4)
        x2 = torch.empty(0, 5)

        expected = [0, 5]
        p1 = pad(x1, expected)
        p2 = pad(x2, expected)
        expected = tuple(expected)
        assert p1.shape == expected
        assert p2.shape == expected

    def test_padded_stack_zero_dim(self):
        torch.random.manual_seed(42)

        x1 = torch.rand(10, 4)
        x2 = torch.rand(0, 4)

        result = padded_stack([x1, x2], pad_value=-1, dim=0)
        assert result.shape == (2, 10, 4)
        assert (result[1] == -1).all()

    @pytest.mark.parametrize("pad_val", [0, float("nan")])
    def test_trim_padding(self, pad_val):
        torch.random.manual_seed(42)
        x = torch.empty(1, 32, 32).fill_(pad_val)
        x[..., :10, :10] = torch.ones(1, 10, 10)

        result = trim_padding(x, pad_value=pad_val)
        assert result.shape == (1, 10, 10)
        assert (result == 1).all()

    def test_detach(self, target_factory):
        dc = target_factory(requires_grad=True)
        out = dc.detach()
        assert isinstance(out, type(dc))
        assert not out.data.requires_grad

    def test_cpu(self, target_factory, cuda):
        dc = target_factory(cuda=cuda)
        out = dc.cpu()
        assert isinstance(out, type(dc))
        assert out.data.device == torch.device("cpu")
        assert torch.allclose(out.data.cpu(), dc.data.cpu())

    def test_to(self, target_factory, cuda):
        dc = target_factory(cuda=cuda)
        out = dc.to("cpu")
        assert isinstance(out, type(dc))
        assert out.data.device == torch.device("cpu")
        assert torch.allclose(out.data.cpu(), dc.data.cpu())

    @cuda_or_skip
    def test_chain(self, target_factory):
        dc = target_factory()
        dc.data.requires_grad = True
        dc.data = dc.data.to("cuda:0")
        out1 = dc.detach().cpu()
        out2 = dc.cpu().detach()
        for out in out1, out2:
            assert out.data.device == torch.device("cpu")
            assert not out.data.requires_grad
