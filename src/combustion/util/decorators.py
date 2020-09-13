#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional, Tuple

import torch
from decorator import decorator, getfullargspec


def input(
    arg: str,
    name: Optional[Tuple[str, ...]] = None,
    shape: Optional[Tuple[Optional[int], ...]] = None,
    drop_names: bool = False,
    optional: bool = False,
):
    """input Decorator for validating / coercing input tensors by parameter
    name.

    :param arg: The name of the parameter to be validated :type arg: str
    :param name: Names to use for NamedTensor creation :type name:
    Optional[Tuple[str, ...]] :param shape: Expected shape of the input
    :type shape: Optional[Tuple[Optional[int], ...]] :param drop_names:
    If true, drop names from the input after validation :type
    drop_names: bool :param optional: If true, no errors will be raised
    if arg is not passed :type drop_names: bool
    """
    if name is None and shape is None:
        raise ValueError("shape or name must be supplied")

    def caller(f, *args, **kwargs):
        spec = getfullargspec(f)
        if arg not in spec.args + spec.kwonlyargs:
            raise ValueError("in_name arg %s not in spec %s" % (arg, spec.args + spec.kwonlyargs))
        arg_pos = spec.args.index(arg) if arg in spec.args else None
        return __in_name(f, arg, arg_pos, name, shape, drop_names, optional, *args, **kwargs)

    return decorator(caller)


def output(
    pos: int = 0,
    name: Optional[Tuple[str, ...]] = None,
    shape: Optional[Tuple[Optional[int], ...]] = None,
):
    """output Decorator for validating / coercing output tensors by position.

    :param pos: Position of the output to be validated if multiple
    return values :type pos: int :param name: Names to apply to output
    :type name: Optional[Tuple[str, ...]] :param shape: Expected shape
    of the output :type shape: Optional[Tuple[Optional[int], ...]]
    """
    if name is None and shape is None:
        raise ValueError("shape or name must be supplied")

    def caller(f, *args, **kwargs):
        return __out_name(f, pos, name, shape, *args, **kwargs)

    return decorator(caller)


def __out_name(f, pos, name, shape, *args, **kwargs):
    result = f(*args, **kwargs)
    if not isinstance(result, tuple):
        result = (result,)
    result = list(result)
    target = result[pos]
    if not isinstance(target, torch.Tensor):
        raise ValueError("expected Tensor at returned index %d, found %s" % (pos, type(target)))
    if name is not None:
        target = __try_add_names(pos, target, *name)
    if shape is not None:
        __check_shape_match(pos, target, shape)
    result[pos] = target
    return tuple(result) if len(result) > 1 else result[0]


def __in_name(f, arg, arg_pos, name, shape, drop_names, optional, *args, **kwargs):
    if arg in kwargs.keys() and kwargs[arg] is not None:
        if name is not None:
            kwargs[arg] = __try_add_names(arg, kwargs[arg], *name)
            if drop_names:
                kwargs[arg] = kwargs[arg].rename(None)
        if shape is not None:
            __check_shape_match(arg, kwargs[arg], shape)
    elif arg_pos is not None and arg_pos < len(args) and args[arg_pos] is not None:
        _ = list(args)
        if name is not None:
            _[arg_pos] = __try_add_names(arg, _[arg_pos], *name)
            if drop_names:
                _[arg_pos] = _[arg_pos].rename(None)
        if shape is not None:
            __check_shape_match(arg, _[arg_pos], shape)
        args = tuple(_)
    elif not optional:
        raise ValueError("expected value for non-optional input %s" % arg)
    return f(*args, **kwargs)


def __try_add_names(arg, tensor, *names):
    param_type = "returned val" if isinstance(arg, int) else "arg"

    if all(tensor.names):
        tensor = tensor.align_to(*names)
    else:
        try:
            tensor = tensor.refine_names(*names)
        except RuntimeError:
            raise ValueError(f"{param_type} {arg}: could not apply names {names} to {tensor.names}")
    return tensor


def __check_shape_match(arg, tensor, shape):
    if isinstance(arg, int):
        param_type = "returned val"
    else:
        param_type = "arg"
    expected_ndim = len(shape)

    if not tensor.ndim == expected_ndim:
        raise ValueError(f"{param_type} {arg}: expected ndim {expected_ndim}, got {tensor.ndim}")
    for i, (expected, actual) in enumerate(zip(shape, tensor.shape)):
        if expected is not None and expected != actual:
            raise ValueError(f"{param_type} {arg}: shape mismatch in dim {i}: expected {expected}, got {actual}")
