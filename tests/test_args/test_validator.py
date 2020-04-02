#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

import pytest

from combustion.args import NumericValidator


class TestNumericValidatorConstructor:
    @pytest.fixture(
        scope="class",
        params=[
            pytest.param(int, id="dtype=int"),
            pytest.param(float, id="dtype=float"),
            pytest.param(str, id="dtype=str", marks=pytest.mark.xfail(strict=True, raises=TypeError)),
        ],
    )
    def dtype(self, request):
        return request.param

    @pytest.fixture(
        scope="class",
        params=[pytest.param(0, id="low=0"), pytest.param(-10, id="low=-10"), pytest.param(None, id="low=None"),],
    )
    def low(self, request):
        return request.param

    @pytest.fixture(
        params=[
            pytest.param(0, id="high=0"),
            pytest.param(-10, id="high=-10"),
            pytest.param(10, id="high=10"),
            pytest.param(None, id="high=None"),
        ]
    )
    def high(self, request, low):
        high = request.param
        if high is not None and low is not None and high <= low:
            request.applymarker(pytest.mark.xfail(strict=True, raises=ValueError))
        elif high is None and low is None:
            request.applymarker(pytest.mark.xfail(strict=True, raises=ValueError))
        return high

    @pytest.fixture(
        scope="class",
        params=[
            pytest.param(True, id="inclusive=True"),
            pytest.param(False, id="inclusive=False"),
            pytest.param((True, False), id="inclusive=(True,False)"),
            pytest.param((False, True), id="inclusive=(False,True)"),
        ],
    )
    def inclusive(self, request):
        return request.param

    def test_constructor_dtype(self, parser, dtype):
        v = NumericValidator(["foo"], "foo", type=dtype, low=0)
        assert v.type == dtype

    def test_constructor_range(self, parser, low, high):
        v = NumericValidator(["foo"], "foo", type=int, low=low, high=high,)

        if low is None:
            assert v.low == float("-inf")
        else:
            assert v.low == low

        if high is None:
            assert v.high == float("inf")
        else:
            assert v.high == high

    def test_constructor_inclusive(self, parser, inclusive):
        v = NumericValidator(["foo"], "foo", type=int, low=0, high=10, inclusive=inclusive)
        if isinstance(inclusive, tuple):
            assert v.inclusive == inclusive
        else:
            assert v.inclusive == (inclusive,) * 2

    @pytest.mark.parametrize(
        "dtype,low,high",
        [
            pytest.param(int, 0, 10, id="case1"),
            pytest.param(int, 0.8, 10, id="case2", marks=pytest.mark.xfail(strict=True)),
            pytest.param(int, 0, 10.5, id="case3", marks=pytest.mark.xfail(strict=True)),
            pytest.param(float, int(0), int(10), id="case4"),
        ],
    )
    @pytest.mark.skip
    def test_constructor_range_type_convert(self, parser, dtype, low, high):
        v = NumericValidator(["foo"], "foo", type=int, low=low, high=high,)
        assert isinstance(v.low, dtype) or v.low == float("-inf")
        assert isinstance(v.high, dtype) or v.high == float("inf")


class TestNumericValidatorCall:
    @pytest.fixture(scope="class", params=[pytest.param(int, id="dtype=int"), pytest.param(float, id="dtype=float"),])
    def dtype(self, request):
        return request.param

    @pytest.fixture(scope="class", params=[pytest.param(i, id="low=%d" % i) for i in range(-10, 11, 8)])
    def low(self, request, dtype):
        return dtype(request.param)

    @pytest.fixture(params=[pytest.param(i, id="high_delt=%d" % i) for i in range(1, 23, 10)])
    def high(self, request, low):
        return request.param + low

    @pytest.fixture(
        scope="class",
        params=[
            pytest.param(True, id="inclusive=T"),
            pytest.param(False, id="inclusive=F"),
            pytest.param((True, False), id="inclusive=(T,F)"),
            pytest.param((False, True), id="inclusive=(F,T)"),
        ],
    )
    def inclusive(self, request):
        return request.param

    @pytest.fixture(params=[pytest.param(x, id="val={}".format(x)) for x in range(-15, 15, 10)])
    def val(self, request, low, high, inclusive):
        val = request.param
        if type(inclusive) != tuple:
            inclusive = (inclusive,) * 2

        d = val - low
        if (inclusive[0] and d < 0) or (not inclusive[0] and d <= 0):
            request.applymarker(pytest.mark.xfail(strict=True, raises=argparse.ArgumentError))

        d = high - val
        if (inclusive[1] and d < 0) or (not inclusive[1] and d <= 0):
            request.applymarker(pytest.mark.xfail(strict=True, raises=argparse.ArgumentError))
        return val

    @pytest.fixture
    def validator(self, dtype, low, high, inclusive, mock_parser):
        v = NumericValidator(["foo"], "foo", type=dtype, low=low, high=high, inclusive=inclusive)
        ns = argparse.Namespace()

        def call(val):
            v(mock_parser, ns, val)
            return ns.foo

        return call

    def test_call(self, val, validator):
        assert validator(val) == val
        assert validator([val, val]) == [val, val]


class TestNumericValidatorIntegration:
    @pytest.fixture
    def add_argument(self, parser):
        parser.add_argument("--foo", type=int, low=0, high=10, inclusive=(True, True), action=NumericValidator)
        return parser

    @pytest.fixture(params=[pytest.param(i, id="val=%d" % i) for i in range(-5, 15, 4)])
    def val(self, request):
        val = request.param
        if val < 0 or val > 10:
            request.applymarker(pytest.mark.xfail(strict=True))
        return val

    @pytest.mark.usefixtures("add_argument")
    def test_call(self, val, parser):
        args = parser.parse_args(args=["--foo", str(val)])
        assert args.foo == val
