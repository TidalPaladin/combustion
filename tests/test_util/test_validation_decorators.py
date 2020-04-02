#!/usr/bin/env python
# -*- coding: utf-8 -*-

from inspect import signature

import pytest

from combustion.util.pytorch import input, output


@pytest.fixture
def func(mocker, torch):
    def f(arg1, arg2=None, *args, **kwargs):
        """foo bar baz"""
        pass

    m = mocker.create_autospec(f, spec_set=True)
    return m


def assert_is_unnamed(tensor):
    __tracebackhide__ = True
    if any(tensor.names):
        pytest.fail("Expected unnamed tensor, but got names %s" % str(tensor.names))


def assert_has_names(tensor, names):
    __tracebackhide__ = True
    if tensor.names != names:
        pytest.fail("Expected names %s, but got names %s" % (names, tensor.names))


class TestInput:
    def test_calls_original_func(self, func, torch):
        tensor = torch.ones(10)
        decorated = input("arg1", shape=(tensor.shape))(func)
        decorated(tensor)
        func.assert_called_once()

    @pytest.mark.parametrize(
        "shape,names", [pytest.param((4,), ("A",), id="dim=1"), pytest.param((4, 5), ("A", "B"), id="dim=2"),]
    )
    def test_drop_names(self, func, shape, names, torch):
        tensor = torch.ones(*shape, names=names)
        decorated = input("arg1", name=names, drop_names=True)(func)
        decorated(tensor)
        func.assert_called_once()
        assert_is_unnamed(func.call_args[0][0])

    @pytest.mark.parametrize(
        "shape,names_in,names_out",
        [
            pytest.param((4,), (None,), ("A",), id="None->A"),
            pytest.param((4,), ("A",), ("A",), id="A->A"),
            pytest.param((4, 5), (None, "B"), ("A", "B"), id="None,B->A,B"),
            pytest.param((4, 5), (None, None), ("A", "B"), id="None,None->A,B"),
            pytest.param(
                (4, 5), (None, "C"), ("A", "B"), marks=pytest.mark.xfail(raises=ValueError), id="None,C->A,B"
            ),
        ],
    )
    def test_coerce_named_input(self, func, shape, names_in, names_out, torch):
        tensor = torch.ones(*shape, names=names_in)
        decorated = input("arg1", name=names_out)(func)
        decorated(tensor)
        func.assert_called_once()
        assert_has_names(func.call_args[0][0], names_out)

    @pytest.mark.parametrize(
        "pre, post",
        [
            pytest.param((4,), (4,), id="4,->4,"),
            pytest.param((None,), (4,), id="None,->4,"),
            pytest.param((4,), (3,), marks=pytest.mark.xfail(raises=ValueError), id="4,->3,"),
            pytest.param((4, 5), (4, 5), id="4,5->4,5"),
            pytest.param((4, 5), (4, 4), marks=pytest.mark.xfail(raises=ValueError), id="4,5->4,4"),
            pytest.param((4, 5), (5, 4), marks=pytest.mark.xfail(raises=ValueError), id="4,5->5,4"),
            pytest.param((None, 5), (4, 5), id="None,5->4,5"),
            pytest.param((None, None), (4, 4), id="None,None->4,4"),
            pytest.param((None, 5), (5, 4), marks=pytest.mark.xfail(raises=ValueError), id="None,5->None,4"),
            pytest.param((4, None), (5, 4), marks=pytest.mark.xfail(raises=ValueError), id="4,None->5,4"),
        ],
    )
    def test_validates_shape(self, func, pre, post, torch):
        tensor = torch.ones(*post)
        decorated = input("arg1", shape=pre)(func)
        decorated(tensor)
        func.assert_called_once()
        assert func.call_args[0][0].shape == post

    def test_selects_correct_arg_by_name(self, func, torch):
        arg1 = torch.ones(10)
        arg2 = torch.ones(10)
        added_name = ("A",)
        decorated = input("arg1", name=added_name)(func)
        decorated(arg1=arg1, arg2=arg2)
        func.assert_called_once()
        assert_has_names(func.call_args[0][0], added_name)
        assert_is_unnamed(func.call_args[0][1])

    def test_preserves_signature(self, func):
        decorated = input("arg1", name=("A",))(func)
        assert signature(decorated) == signature(func)

    def test_preserves_docstring(self, func):
        def f(arg1, arg2=None, *args, **kwargs):
            """foo bar baz"""
            pass

        decorated = input("arg1", name=("A",))(f)
        assert decorated.__doc__
        assert decorated.__doc__ == f.__doc__

    def test_multiple_decorators(self, func, torch):
        args = [torch.ones(10), torch.ones(10)]
        _ = input("arg1", shape=(args[0].shape), name=("A",))(func)
        decorated = input("arg2", shape=(args[1].shape), name=("B",))(_)
        decorated(arg1=args[0], arg2=args[1])
        func.assert_called_once()
        assert_has_names(func.call_args[0][0], ("A",))
        assert_has_names(func.call_args[0][1], ("B",))

    @pytest.mark.parametrize("optional", [True, False])
    @pytest.mark.parametrize("arg_given", [True, False])
    def test_optional(self, func, torch, optional, arg_given):
        decorated = input("arg2", shape=(None,), optional=optional)(func)
        if not optional and not arg_given:
            with pytest.raises(ValueError):
                decorated(arg1=torch.ones(10))
        else:
            if arg_given:
                decorated(arg1=torch.ones(10), arg2=torch.ones(10))
            else:
                decorated(arg1=torch.ones(10))
            func.assert_called()

    def test_kw_only_args(self, torch):
        def f(*, arg1, arg2=None, **kwargs):
            """foo bar baz"""
            pass

        decorated = input("arg1", name=("A",))(f)
        decorated(arg1=torch.ones(10))


class TestOutput:
    def test_calls_original_func(self, func, torch):
        tensor = torch.ones(10)
        func.return_value = tensor
        decorated = output(shape=(tensor.shape))(func)
        decorated(tensor)
        func.assert_called_once()

    @pytest.mark.parametrize(
        "shape,names_in,names_out",
        [
            pytest.param((4,), (None,), ("A",), id="None->A"),
            pytest.param((4,), ("A",), ("A",), id="A->A"),
            pytest.param((4, 5), (None, "B"), ("A", "B"), id="None,B->A,B"),
            pytest.param((4, 5), (None, None), ("A", "B"), id="None,None->A,B"),
            pytest.param(
                (4, 5), (None, "C"), ("A", "B"), marks=pytest.mark.xfail(raises=ValueError), id="None,C->A,B"
            ),
        ],
    )
    def test_coerce_named_output(self, shape, names_in, names_out, torch):
        tensor = torch.ones(*shape, names=names_in)

        def f():
            return tensor

        decorated = output(name=names_out)(f)
        result = decorated()
        assert_has_names(result, names_out)

    @pytest.mark.parametrize(
        "pre, post",
        [
            pytest.param((4,), (4,), id="4,->4,"),
            pytest.param((None,), (4,), id="None,->4,"),
            pytest.param((4,), (3,), marks=pytest.mark.xfail(raises=ValueError), id="4,->3,"),
            pytest.param((4, 5), (4, 5), id="4,5->4,5"),
            pytest.param((4, 5), (4, 4), marks=pytest.mark.xfail(raises=ValueError), id="4,5->4,4"),
            pytest.param((4, 5), (5, 4), marks=pytest.mark.xfail(raises=ValueError), id="4,5->5,4"),
            pytest.param((None, 5), (4, 5), id="None,5->4,5"),
            pytest.param((None, None), (4, 4), id="None,None->4,4"),
            pytest.param((None, 5), (5, 4), marks=pytest.mark.xfail(raises=ValueError), id="None,5->None,4"),
            pytest.param((4, None), (5, 4), marks=pytest.mark.xfail(raises=ValueError), id="4,None->5,4"),
        ],
    )
    def test_validates_shape(self, pre, post, torch):
        tensor = torch.ones(*post)

        def f():
            return tensor

        decorated = output(shape=pre)(f)
        result = decorated()
        assert result.shape == post

    @pytest.mark.parametrize(
        "pos",
        [
            pytest.param(0, id="pos=0"),
            pytest.param(1, id="pos=1"),
            pytest.param(2, id="oob", marks=pytest.mark.xfail(raises=IndexError)),
        ],
    )
    def test_selects_correct_arg_by_pos(self, pos, torch):
        args = [torch.ones(10), torch.ones(10)]

        def f():
            return args[0], args[1]

        added_name = ("A",)
        decorated = output(pos, name=added_name)(f)
        result = list(decorated())
        assert_has_names(result[pos], added_name)
        del result[pos]
        assert_is_unnamed(result[0])

    def test_preserves_signature(self):
        def f(arg1, arg2=None, *args, **kwargs):
            pass

        decorated = output(name=("A",))(f)
        assert signature(decorated) == signature(f)

    def test_preserves_docstring(self):
        def f(arg1, arg2=None, *args, **kwargs):
            """foo bar baz"""
            pass

        decorated = output(name=("A",))(f)
        assert decorated.__doc__
        assert decorated.__doc__ == f.__doc__

    def test_multiple_decorators(self, func, torch):
        ret = (torch.ones(10), torch.ones(10))
        func.return_value = ret
        _ = output(0, shape=(ret[0].shape), name=("A",))(func)
        decorated = output(1, shape=(ret[0].shape), name=("B",))(_)
        result = decorated(torch.ones(10))
        func.assert_called_once()
        assert_has_names(result[0], ("A",))
        assert_has_names(result[1], ("B",))
