#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from functools import partial
from typing import Any, Generic, Optional, Sequence, Type, TypeVar, Union

import pytest
import torch
from torch import Tensor
from torch.testing import assert_allclose

from combustion.vision.coords import BoundingBox2d, Coordinates


U = TypeVar("U", bound="Coordinates")


class TestCoordinates(Generic[U]):
    TEST_CLS: Type[U] = Coordinates  # type: ignore

    @pytest.fixture(params=[2, 3])
    def coord_dims(self, request):
        return request.param

    @pytest.fixture
    def coords_factory(self, padded_coords_factory):
        def func(
            traces: int = 3,
            trace_len: int = 10,
            batch_size: Optional[int] = None,
            seed: int = 42,
            pad_val: Any = -1,
            cuda: bool = False,
            requires_grad: bool = False,
            lower_bound: Union[float, Sequence[float]] = 0.0,
            upper_bound: Union[float, Sequence[float]] = 1.0,
            coord_dims: int = 2,
        ) -> U:
            coords, padding = padded_coords_factory(
                traces, trace_len, batch_size, seed, pad_val, cuda, requires_grad, lower_bound, upper_bound, coord_dims
            )
            return self.TEST_CLS.from_padded(coords, pad_val)

        return func

    @pytest.mark.parametrize("trace_len", [5, 10])
    @pytest.mark.parametrize("traces", [1, 3])
    @pytest.mark.parametrize("requires_grad", [False, True])
    @pytest.mark.parametrize("pad_val", [-1, -2])
    @pytest.mark.parametrize("batch_size", [None, 2])
    @pytest.mark.parametrize("seed", [42, 43])
    def test_from_padded(
        self, padded_coords_factory, pad_val, batch_size, seed, cuda, requires_grad, traces, trace_len, coord_dims
    ):
        coords, padding = padded_coords_factory(
            traces=traces,
            trace_len=trace_len,
            batch_size=batch_size,
            seed=seed,
            pad_val=pad_val,
            cuda=cuda,
            requires_grad=requires_grad,
            coord_dims=coord_dims,
        )

        struct = Coordinates.from_padded(coords, pad_val)
        shape = struct.coords.shape
        if batch_size:
            assert shape == (batch_size, traces, trace_len, coord_dims)
        else:
            assert shape == (traces, trace_len, coord_dims)
        assert_allclose(struct.coords.values(), coords[~padding])
        assert struct.coords.device == coords.device
        assert struct.coords.requires_grad == requires_grad

    @pytest.mark.parametrize("pad_val", [-1, -2])
    @pytest.mark.parametrize("batch_size", [None, 2])
    @pytest.mark.parametrize("seed", [42, 43])
    @pytest.mark.parametrize("requires_grad", [False, True])
    @pytest.mark.parametrize("op", [torch.add, torch.sub])
    def test_torch_function_coords(
        self, coords_factory, pad_val, batch_size, seed, cuda, requires_grad, op, coord_dims
    ):
        coords = coords_factory(
            batch_size=batch_size,
            seed=seed,
            pad_val=pad_val,
            cuda=cuda,
            requires_grad=requires_grad,
            coord_dims=coord_dims,
        )
        output = op(coords, coords)  # type: ignore
        expected = op(coords.coords, coords.coords).coalesce()
        expected = Coordinates(output.sparse_mask(expected.to_dense()))
        assert isinstance(output, Coordinates)
        assert output.coords.is_coalesced()
        assert_allclose(output.coords, expected.coords)
        assert output.coords.shape == coords.coords.shape
        assert output.coords.requires_grad == requires_grad
        assert output.coords.device == coords.coords.device

    @pytest.mark.parametrize("pad_val", [-1, -2])
    @pytest.mark.parametrize("batch_size", [None, 2])
    @pytest.mark.parametrize("seed", [42, 43])
    @pytest.mark.parametrize("requires_grad", [False, True])
    @pytest.mark.parametrize("op", [torch.add, torch.sub, torch.mul, torch.div, torch.clamp_max, torch.clamp_min])
    @pytest.mark.parametrize("offset", [torch.tensor(1), torch.tensor(2)])
    @pytest.mark.parametrize("sparse", [False, True])
    def test_torch_function_tensor(
        self, coords_factory, pad_val, batch_size, seed, cuda, requires_grad, op, offset, sparse, coord_dims
    ):
        coords = coords_factory(
            batch_size=batch_size,
            seed=seed,
            pad_val=pad_val,
            cuda=cuda,
            requires_grad=requires_grad,
            coord_dims=coord_dims,
        )
        offset = offset.to(coords.coords.device)
        if sparse:
            offset = offset.to_sparse()

        output = op(coords, offset)  # type: ignore
        expected = op(coords.coords.to_dense(), offset.to_dense() if offset.is_sparse else offset).sparse_mask(
            coords.coords
        )
        assert isinstance(output, Coordinates)
        assert output.coords.is_coalesced()
        assert_allclose(output.coords, expected)
        assert output.coords.shape == coords.coords.shape
        assert output.coords.requires_grad == requires_grad
        assert output.coords.device == coords.coords.device

    @pytest.mark.parametrize("batch_size", [None, 2])
    @pytest.mark.parametrize("seed", [42, 43])
    def test_sparse_mask(self, coords_factory, batch_size, seed, cuda):
        coords = coords_factory(
            batch_size=batch_size,
            seed=seed,
            cuda=cuda,
        )
        t = coords.coords.to_dense()
        output = coords.sparse_mask(t)
        assert_allclose(output, coords.coords)

    def test_shape(self, coords_factory):
        default_coords = coords_factory()
        assert default_coords.shape == default_coords.coords.shape

    def test_size(self, coords_factory):
        default_coords = coords_factory()
        assert default_coords.size() == default_coords.coords.size()

    @pytest.mark.parametrize("batch_size", [None, 2])
    def test_len(self, coords_factory, batch_size):
        coords = coords_factory(batch_size=batch_size)
        assert len(coords) == len(coords.coords)

    def test_repr(self, coords_factory):
        coords = coords_factory()
        s = str(coords)
        assert isinstance(s, str)

    @pytest.mark.parametrize("batch_size", [None, 2, 4])
    def test_getitem(self, coords_factory, batch_size):
        coords = coords_factory(batch_size=batch_size)
        sliced = coords[0]
        assert isinstance(sliced, Coordinates)
        assert sliced.ndim == max(coords.ndim - 1, 2)

    @pytest.mark.parametrize("requires_grad", [False, True])
    def test_hash(self, coords_factory, cuda, requires_grad):
        factory = partial(coords_factory, requires_grad=requires_grad, cuda=cuda)
        coords1 = factory(seed=1, lower_bound=0, upper_bound=1)
        coords2 = factory(seed=2, lower_bound=-1, upper_bound=2)
        coords3 = factory(seed=2, lower_bound=-1, upper_bound=2)
        assert hash(coords1) != hash(coords2)
        assert hash(coords2) == hash(coords3)

    @pytest.mark.parametrize("requires_grad", [False, True])
    def test_eq(self, coords_factory, cuda, requires_grad):
        factory = partial(coords_factory, requires_grad=requires_grad, cuda=cuda)
        coords1 = factory(seed=1, lower_bound=0, upper_bound=1)
        coords2 = factory(seed=2, lower_bound=-1, upper_bound=2)
        coords3 = factory(seed=2, lower_bound=-1, upper_bound=2)
        assert coords1 != coords2
        assert coords2 == coords3

    @pytest.mark.parametrize(
        "low,high,expected",
        [
            pytest.param(0.0, 1.0, True),
            pytest.param(1.0, 2.0, False),
            pytest.param(-1.0, 0.0, False),
        ],
    )
    def test_is_fractional(self, coords_factory, low, high, expected):
        coords = coords_factory(lower_bound=low, upper_bound=high)
        assert coords.is_fractional == expected

    @pytest.mark.parametrize(
        "coord_dims,size",
        [
            pytest.param(1, (10,)),
            pytest.param(2, (10, 10)),
            pytest.param(2, (10, 20)),
            pytest.param(2, (20, 10)),
            pytest.param(3, (10, 10, 10)),
            pytest.param(3, (10, 10, 20)),
        ],
    )
    def test_to_fractional(self, coords_factory, coord_dims, size, cuda):
        coords = coords_factory(coord_dims=coord_dims, upper_bound=size, cuda=cuda)
        actual = coords.to_fractional(size)
        assert (actual.coords.values() >= 0).all()
        assert (actual.coords.values() <= 1.0).all()
        expected = coords.sparse_mask(coords.coords.to_dense() / coords.coords.new_tensor(size))
        assert_allclose(actual.coords, expected)

    @pytest.mark.parametrize(
        "coord_dims,size",
        [
            pytest.param(1, (10,)),
            pytest.param(2, (10, 10)),
            pytest.param(2, (10, 20)),
            pytest.param(2, (20, 10)),
            pytest.param(3, (10, 10, 10)),
            pytest.param(3, (10, 10, 20)),
        ],
    )
    def test_from_fractional(self, coords_factory, coord_dims, size, cuda):
        coords = coords_factory(coord_dims=coord_dims, cuda=cuda)
        actual = coords.from_fractional(size)
        expected = coords.sparse_mask(coords.coords.to_dense() * coords.coords.new_tensor(size))
        assert_allclose(actual.coords, expected)

    @pytest.mark.parametrize(
        "coord_dims,size",
        [
            pytest.param(1, (10,)),
            pytest.param(2, (10, 10)),
            pytest.param(2, (10, 20)),
            pytest.param(2, (20, 10)),
            pytest.param(3, (10, 10, 10)),
            pytest.param(3, (10, 10, 20)),
        ],
    )
    def test_clip_to_size(self, coords_factory, coord_dims, size, cuda):
        coords = coords_factory(lower_bound=-100, upper_bound=100, coord_dims=coord_dims, cuda=cuda)
        actual = coords.clip_to_size(size)
        assert (actual.coords.values() >= 0).all()
        for i, bound in enumerate(size):
            assert (actual.coords.values()[..., i] <= bound).all()

    @pytest.mark.parametrize(
        "coord_dims,lower_bound,upper_bound",
        [
            pytest.param(2, (0, 0), (10, 10)),
        ],
    )
    def test_to_box(self, coords_factory, coord_dims, lower_bound, upper_bound, cuda):
        coords = coords_factory(lower_bound=lower_bound, upper_bound=upper_bound, coord_dims=coord_dims, cuda=cuda)
        box = coords.box
        assert isinstance(box, BoundingBox2d)
        assert box.coords.shape[-2] == 2**coord_dims

    @pytest.mark.parametrize("pad_val", [-1, -2])
    @pytest.mark.parametrize("seed", [42, 43])
    @pytest.mark.parametrize("requires_grad", [False, True])
    def test_from_unbatched(self, coords_factory, pad_val, seed, cuda, requires_grad, coord_dims):
        coords = coords_factory(
            seed=seed,
            pad_val=pad_val,
            cuda=cuda,
            requires_grad=requires_grad,
            coord_dims=coord_dims,
        )
        output = self.TEST_CLS.from_unbatched([coords, coords])
        assert output.is_batched
        assert len(output) == 2
        assert output[0] == coords
        assert output[1] == coords
        assert output.coords.device == coords.coords.device


@dataclass(frozen=True, repr=False, eq=False)
class CoordinatesWithAttributes(Coordinates):
    labels: Optional[Tensor] = None
    __slice_fields__ = ["coords", "labels"]


class TestCoordinatesWithAttributes(TestCoordinates):
    TEST_CLS = CoordinatesWithAttributes

    @pytest.fixture(params=[True, False])
    def coords_factory(self, request, padded_coords_factory):
        def func(
            traces: int = 3,
            trace_len: int = 10,
            batch_size: Optional[int] = None,
            seed: int = 42,
            pad_val: Any = -1,
            cuda: bool = False,
            requires_grad: bool = False,
            lower_bound: Union[float, Sequence[float]] = 0.0,
            upper_bound: Union[float, Sequence[float]] = 1.0,
            coord_dims: int = 2,
        ) -> Coordinates:
            coords, padding = padded_coords_factory(
                traces, trace_len, batch_size, seed, pad_val, cuda, requires_grad, lower_bound, upper_bound, coord_dims
            )
            labels = coords[..., 0].clone() if request.param else None
            return self.TEST_CLS.from_padded(coords, pad_val, labels=labels)

        return func


class TestBoundingBox2d(TestCoordinates):
    @pytest.fixture(params=[2])
    def coord_dims(self, request):
        return request.param

    @pytest.fixture
    def coords_factory(self, padded_coords_factory):
        def func(
            traces: int = 3,
            trace_len: int = 10,
            batch_size: Optional[int] = None,
            seed: int = 42,
            pad_val: Any = -1,
            cuda: bool = False,
            requires_grad: bool = False,
            lower_bound: Union[float, Sequence[float]] = 0.0,
            upper_bound: Union[float, Sequence[float]] = 1.0,
            coord_dims: int = 2,
        ) -> BoundingBox2d:
            if coord_dims != 2:
                pytest.skip(msg="3d coords not supported")
            coords, padding = padded_coords_factory(
                traces, trace_len, batch_size, seed, pad_val, cuda, requires_grad, lower_bound, upper_bound, coord_dims
            )
            return Coordinates.from_padded(coords, pad_val).box

        return func

    def test_from_padded(self):
        pass

    # @pytest.mark.parametrize("coord_dims,lower_bound,upper_bound", [
    #    pytest.param(2, (0, 0), (10, 10)),
    # ])
    # def test_to_box(self, coords_factory, coord_dims, lower_bound, upper_bound, cuda):
    #    coords = coords_factory(lower_bound=lower_bound, upper_bound=upper_bound, coord_dims=coord_dims, cuda=cuda)
    #    box = coords.box
    #    assert False
