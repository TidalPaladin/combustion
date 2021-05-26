#!/usr/bin/env python
# -*- coding: utf-8 -*-

import types
from typing import Any

import pytest
import torch

from combustion.util import (
    check_dimension,
    check_dimension_match,
    check_dimension_within_range,
    check_is_array,
    check_is_tensor,
    check_ndim,
    check_ndim_match,
    check_ndim_within_range,
    check_shape,
    check_shapes_match,
)


class CaseMeta(type):
    r"""Metaclass to parametrize ``test_raises_exception`` using cases from
    ``get_test_cases``.
    """

    def __new__(cls: Any, name, bases, dct):
        x = super().__new__(cls, name, bases, dct)
        x.test_raises_exception = pytest.mark.parametrize(
            "inputs,exception",
            x.get_test_cases(),
        )(cls.copy_func(bases[0].test_raises_exception))
        return x

    @classmethod
    def copy_func(cls, f, name=None):
        """Used to copy parametrized method so parametrization's dont stack"""
        fn = types.FunctionType(f.__code__, f.__globals__, name or f.__name__, f.__defaults__, f.__closure__)
        # in case f was given attrs (note this dict is a shallow copy):
        fn.__dict__.update(f.__dict__)
        return fn


class BaseValidationTest:

    # TODO handle this with torch.jit.ignore in torch 1.8
    @pytest.fixture(
        params=[
            pytest.param(False, id="non-jit"),
            # pytest.param(True, id="jit"),
        ]
    )
    def scripted(self, request):
        return request.param

    @pytest.fixture
    def func(self):
        raise NotImplementedError("function under test")

    def get_test_cases(self):
        raise NotImplementedError("function inputs and expected outcome")

    def test_raises_exception(self, func, inputs, exception, scripted):
        if scripted:
            func = torch.jit.script(func)

        if exception is not None:
            with pytest.raises(exception):
                func(*inputs)
        else:
            func(*inputs)


class TestCheckShapesMatch(BaseValidationTest, metaclass=CaseMeta):
    @staticmethod
    def get_test_cases():
        cases = [
            pytest.param((torch.rand(10), torch.rand(10), "x", "y"), None, id="matches1"),
            pytest.param((torch.rand(10).numpy(), torch.rand(10).numpy(), "x", "y"), None, id="matches2"),
            pytest.param((torch.rand(10), torch.rand(5), "x", "y"), ValueError, id="mismatch1"),
            pytest.param((torch.rand(10).numpy(), torch.rand(5).numpy(), "x", "y"), ValueError, id="mismatch2"),
        ]
        return cases

    @pytest.fixture
    def func(self):
        return check_shapes_match


class TestCheckNdim(BaseValidationTest, metaclass=CaseMeta):
    @staticmethod
    def get_test_cases():
        torch.random.manual_seed(42)
        cases = [
            pytest.param((torch.rand(1), 1, "x"), None, id="matches1"),
            pytest.param((torch.rand(1, 1), 2, "x"), None, id="matches2"),
            pytest.param((torch.rand(1, 1).numpy(), 2, "x"), None, id="matches3"),
            pytest.param((torch.tensor(1.0), 0, "x"), None, id="matches4"),
            pytest.param((torch.rand(1), 2, "x"), ValueError, id="mismatch1"),
            pytest.param((torch.rand(1, 1), 1, "x"), ValueError, id="mismatch2"),
            pytest.param((torch.rand(1, 1).numpy(), 1, "x"), ValueError, id="mismatch3"),
        ]
        return cases

    @pytest.fixture
    def func(self):
        return check_ndim


class TestCheckNdimMatch(BaseValidationTest, metaclass=CaseMeta):
    @staticmethod
    def get_test_cases():
        cases = [
            pytest.param((torch.rand(10), torch.rand(10), "x", "y"), None, id="matches1"),
            pytest.param((torch.rand(10).numpy(), torch.rand(10).numpy(), "x", "y"), None, id="matches2"),
            pytest.param((torch.rand(10, 1), torch.rand(10), "x", "y"), ValueError, id="mismatch1"),
            pytest.param((torch.rand(10, 1).numpy(), torch.rand(5).numpy(), "x", "y"), ValueError, id="mismatch2"),
        ]
        return cases

    @pytest.fixture
    def func(self):
        return check_ndim_match


class TestCheckNdimWithinRange(BaseValidationTest, metaclass=CaseMeta):
    @staticmethod
    def get_test_cases():
        torch.random.manual_seed(42)
        cases = [
            pytest.param((torch.rand(1), (0, 1), "x"), None, id="matches1"),
            pytest.param((torch.rand(1, 1), (1, None), "x"), None, id="matches2"),
            pytest.param((torch.rand(1, 1).numpy(), (None, 3), "x"), None, id="matches3"),
            pytest.param((torch.tensor(1.0), (None, 2), "x"), None, id="matches4"),
            pytest.param((torch.rand(1), (2, ValueError), "x"), ValueError, id="mismatch1"),
            pytest.param((torch.rand(1, 1), (0, 1), "x"), ValueError, id="mismatch2"),
            pytest.param((torch.rand(1, 1).numpy(), (3, 4), "x"), ValueError, id="mismatch3"),
        ]
        return cases

    @pytest.fixture
    def func(self):
        return check_ndim_within_range


class TestIsTensor(BaseValidationTest, metaclass=CaseMeta):
    @staticmethod
    def get_test_cases():
        cases = [
            pytest.param(("foo", "x"), TypeError, id="foo"),
            pytest.param((torch.rand(10), "x"), None, id="torch.rand()"),
            pytest.param((torch.rand(10).numpy(), "x"), TypeError, id="numpy"),
            pytest.param((None, "x"), TypeError, id="None"),
        ]
        return cases

    @pytest.fixture
    def func(self):
        return check_is_tensor


class TestIsArray(BaseValidationTest, metaclass=CaseMeta):
    @staticmethod
    def get_test_cases():
        cases = [
            pytest.param(("foo", "x"), TypeError, id="foo"),
            pytest.param((torch.rand(10), "x"), None, id="torch.rand()"),
            pytest.param((torch.rand(10).numpy(), "x"), None, id="numpy"),
            pytest.param((None, "x"), TypeError, id="None"),
        ]
        return cases

    @pytest.fixture
    def func(self):
        return check_is_array


class TestCheckShape(BaseValidationTest, metaclass=CaseMeta):
    @staticmethod
    def get_test_cases():
        cases = [
            pytest.param((torch.rand(10), (10,), "x"), None, id="matches1"),
            pytest.param((torch.rand(5, 5), torch.Size((5, 5)), "x"), None, id="matches2"),
            pytest.param((torch.rand(5, 5), (5, None), "x"), None, id="matches3"),
            pytest.param((torch.rand(10).numpy(), (10,), "x"), None, id="matches4"),
            pytest.param((torch.rand(10), (9,), "x"), ValueError, id="mismatch1"),
            pytest.param((torch.rand(5, 5), torch.Size((5, 10)), "x"), ValueError, id="mismatch2"),
        ]
        return cases

    @pytest.fixture
    def func(self):
        return check_shape


class TestCheckDimension(BaseValidationTest, metaclass=CaseMeta):
    @staticmethod
    def get_test_cases():
        torch.random.manual_seed(42)
        base = torch.rand(2, 3, 4)
        cases = [
            pytest.param((base, 0, 2, "x"), None, id="matches1"),
            pytest.param((base, 1, 3, "x"), None, id="matches2"),
            pytest.param((base, -1, 4, "x"), None, id="matches3"),
            pytest.param((base.numpy(), 0, 2, "x"), None, id="matches4"),
            pytest.param((base, 0, 3, "x"), ValueError, id="mismatch1"),
            pytest.param((base, 1, 4, "x"), ValueError, id="mismatch2"),
            pytest.param((base, -1, 5, "x"), ValueError, id="mismatch3"),
            pytest.param((base.numpy(), 0, 3, "x"), ValueError, id="mismatch4"),
        ]
        return cases

    @pytest.fixture
    def func(self):
        return check_dimension


class TestCheckDimensionWithinRange(BaseValidationTest, metaclass=CaseMeta):
    @staticmethod
    def get_test_cases():
        torch.random.manual_seed(42)
        base = torch.rand(2, 3, 4)
        cases = [
            pytest.param((base, 0, (2, 3), "x"), None, id="matches1"),
            pytest.param((base, 1, (2, 3), "x"), None, id="matches2"),
            pytest.param((base, -1, (1, 6), "x"), None, id="matches3"),
            pytest.param((base.numpy(), 0, (2, 3), "x"), None, id="matches4"),
            pytest.param((base, 0, (1, None), "x"), None, id="matches5"),
            pytest.param((base, 0, (3, 4), "x"), ValueError, id="mismatch1"),
            pytest.param((base, 1, (1, 2), "x"), ValueError, id="mismatch2"),
            pytest.param((base, -1, (1, 2), "x"), ValueError, id="mismatch3"),
            pytest.param((base, 0, (3, None), "x"), ValueError, id="mismatch1"),
            pytest.param((base.numpy(), 0, (3, 4), "x"), ValueError, id="mismatch4"),
        ]
        return cases

    @pytest.fixture
    def func(self):
        return check_dimension_within_range


class TestCheckDimensionMatch(BaseValidationTest, metaclass=CaseMeta):
    @staticmethod
    def get_test_cases():
        torch.random.manual_seed(42)
        base = torch.rand(2, 3, 4)
        off_base = torch.rand(3, 4, 5)
        cases = [
            pytest.param((base, base, 0, "x", "y"), None, id="matches1"),
            pytest.param((base, base, 1, "x", "y"), None, id="matches2"),
            pytest.param((base, base, -1, "x", "y"), None, id="matches3"),
            pytest.param((base.numpy(), base.numpy(), 0, "x", "y"), None, id="matches4"),
            pytest.param((base, off_base, 0, "x", "y"), ValueError, id="mismatch1"),
            pytest.param((base, off_base, 1, "x", "y"), ValueError, id="mismatch2"),
            pytest.param((base, off_base, -1, "x", "y"), ValueError, id="mismatch3"),
        ]
        return cases

    @pytest.fixture
    def func(self):
        return check_dimension_match
