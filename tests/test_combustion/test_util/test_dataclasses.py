#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import pytest
import torch
from torch import Tensor

from combustion.testing import cuda_or_skip
from combustion.util.compute import slice_along_dim
from combustion.util.dataclasses import BatchMixin, TensorDataclass


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
class BatchedDC(TensorDataclass, BatchMixin):
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


@dataclass(repr=False)
class NestedBatchedDC(TensorDataclass, BatchMixin):
    tensor: Tensor = torch.rand(B, 3, 32, 32)
    tensor_list: List[Tensor] = field(default_factory=lambda: [torch.rand(B, 1, 16, 16)] * 2)
    tensor_tuple: Tuple[Tensor, ...] = field(default_factory=lambda: (torch.rand(B, 2, 32, 16),) * 3)
    optional_tensor: Optional[Tensor] = None
    string: str = "".join(["x"] * B)
    dc: BatchedDC = BatchedDC(torch.rand(B, 3, 32, 32))
    flat_tensor: Tensor = torch.rand(1).squeeze()

    @classmethod
    def from_unbatched(cls):
        pass

    @property
    def is_batched(self):
        return self.tensor.ndim == 4

    def __len__(self):
        assert self.is_batched
        return self.tensor.shape[0]


@dataclass(repr=False)
class TensorDC(TensorDataclass):
    img: Tensor


@dataclass(repr=False)
class DoubleTensorDC(TensorDataclass):
    img1: Tensor
    img2: Tensor


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
        B = 2
        img = torch.rand(B, 3, 32, 32)
        dc = BatchedDC(img)

        for i in range(B):
            sliced = dc[i]
            assert isinstance(sliced, BatchedDC)
            assert torch.allclose(sliced.img, img[i])
            assert not sliced.is_batched

    def test_detach(self):
        torch.random.manual_seed(42)
        B = 2
        img = torch.rand(B, 3, 32, 32, requires_grad=True)
        dc = TensorDC(img)
        out = dc.detach()
        assert isinstance(out, TensorDC)
        assert not out.img.requires_grad

    def test_cpu(self):
        torch.random.manual_seed(42)
        B = 2
        img = torch.rand(B, 3, 32, 32, requires_grad=True)
        dc = TensorDC(img)
        out = dc.cpu()
        assert isinstance(out, TensorDC)
        assert torch.allclose(out.img, img)

    def test_to(self):
        torch.random.manual_seed(42)
        B = 2
        img = torch.rand(B, 3, 32, 32, requires_grad=True)
        dc = TensorDC(img)
        out = dc.to("cpu")
        assert isinstance(out, TensorDC)
        assert torch.allclose(out.img, img)

    @cuda_or_skip
    def test_chain(self):
        torch.random.manual_seed(42)
        B = 2
        img = torch.rand(B, 3, 32, 32, requires_grad=True, device="cuda:0")
        dc = TensorDC(img)
        out1 = dc.detach().cpu()
        out2 = dc.cpu().detach()
        for out in out1, out2:
            assert out.img.device == torch.device("cpu")
            assert not out.img.requires_grad

    def test_device_cpu(self):
        torch.random.manual_seed(42)
        B = 2
        dev1 = dev2 = "cpu"
        exp = "cpu"
        img = torch.rand(B, 3, 32, 32, device=dev1)
        img2 = torch.rand(B, 3, 32, 32, device=dev2)

        dc = DoubleTensorDC(img, img2)
        assert dc.device == torch.device(exp)

    @cuda_or_skip
    @pytest.mark.parametrize(
        "dev1,dev2,exp",
        [
            pytest.param("cpu", "cpu", "cpu"),
            pytest.param("cpu", "cuda:0", None),
            pytest.param("cuda:0", "cpu", None),
            pytest.param("cuda:0", "cuda:0", "cuda:0"),
        ],
    )
    def test_device(self, dev1, dev2, exp):
        torch.random.manual_seed(42)
        B = 2
        img = torch.rand(B, 3, 32, 32, device=dev1)
        img2 = torch.rand(B, 3, 32, 32, device=dev2)

        dc = DoubleTensorDC(img, img2)
        assert dc.device == (torch.device(exp) if exp is not None else None)

    @pytest.mark.parametrize(
        "grad1,grad2,exp",
        [
            pytest.param(True, True, True),
            pytest.param(False, False, False),
            pytest.param(True, False, True),
            pytest.param(False, True, True),
        ],
    )
    def test_requires_grad(self, grad1, grad2, exp):
        torch.random.manual_seed(42)
        B = 2
        img = torch.rand(B, 3, 32, 32, requires_grad=grad1)
        img2 = torch.rand(B, 3, 32, 32, requires_grad=grad2)

        dc = DoubleTensorDC(img, img2)
        assert dc.requires_grad == exp

    def test_getitem_nested(self):
        dc = NestedBatchedDC()

        for idx in range(len(dc)):
            expected = NestedBatchedDC(
                dc.tensor[idx],
                [t[idx] for t in dc.tensor_list],
                tuple([t[idx] for t in dc.tensor_tuple]),
                None,
                dc.string,
                dc.dc[idx],
                dc.flat_tensor,
            )
            sliced = dc[idx]
            assert torch.allclose(sliced.tensor, expected.tensor)
            # TODO test other fields once apply_to_collections supports dataclasses
