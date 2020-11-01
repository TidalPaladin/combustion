#!/usr/bin/env python
# -*- coding: utf-8 -*-


from typing import Optional, Tuple

from decorator import decorator, getfullargspec


def recurse_on_batch(
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
