#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dataclasses import dataclass

import pytest
import torch
from torch import Tensor

from combustion.util.compute import slice_along_dim
from combustion.util.dataclasses import BatchMixin


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


class TestBatchMixin:
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

        result = BatchMixin.padded_stack(tensors, value=-1, dim=0)
        assert result.shape

        assert result.ndim == ndim + 1
        assert result.shape[dim + 1] == max_len
        assert torch.allclose(result[longest_index], max_tensor)

    def test_pad(self):
        torch.random.manual_seed(42)
        x1 = torch.rand(1, 32, 8)
        x2 = torch.rand(3, 18, 19)

        expected = [3, 32, 19]
        p1 = BatchMixin.pad(x1, expected)
        p2 = BatchMixin.pad(x2, expected)
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
        p1 = BatchMixin.pad(x1, expected)
        p2 = BatchMixin.pad(x2, expected)
        expected = tuple(expected)
        assert p1.shape == expected
        assert p2.shape == expected

    def test_padded_stack_zero_dim(self):
        torch.random.manual_seed(42)

        x1 = torch.rand(10, 4)
        x2 = torch.rand(0, 4)

        result = BatchMixin.padded_stack([x1, x2], value=-1, dim=0)
        assert result.shape == (2, 10, 4)
        assert (result[1] == -1).all()

    def test_unpad(self):
        torch.random.manual_seed(42)
        x = torch.zeros(1, 32, 32)
        x[..., :10, :10] = torch.ones(1, 10, 10)

        result = BatchMixin.unpad(x, value=0)
        assert result.shape == (1, 10, 10)
        assert (result == 1).all()

    def test_getitem(self):
        torch.random.manual_seed(42)

        @dataclass
        class SimpleDC(BatchMixin):
            __slice_fields__ = ["img"]
            img: Tensor

            @classmethod
            def from_unbatched(cls):
                pass

            @property
            def is_batched(self):
                return self.img.ndim == 4

            def __len__(self):
                assert self.is_batched
                return self.img.shape[0]

        B = 2
        img = torch.rand(B, 3, 32, 32)
        dc = SimpleDC(img)

        for i in range(B):
            sliced = dc[i]
            assert isinstance(sliced, SimpleDC)
            assert torch.allclose(sliced.img, img[i])
            assert not sliced.is_batched
