#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dataclasses import dataclass, is_dataclass, make_dataclass, field
from enum import Enum
from inspect import signature, Parameter, getmro
from typing import Any, Optional, Tuple, Union, Iterable, Dict, List, Protocol, runtime_checkable
import types

import torch
from decorator import decorator, getfullargspec
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, ListConfig

from abc import ABC


class Instantiable(Protocol):
    _target_: str = MISSING


def hydra_dataclass(
    spec: Optional[type] = None,
    target: Optional[str] = None,
    name: Optional[str] = None,
    group: Optional[str] = None,
    recursive: bool = True,
    overrides: Dict[str, Any] = {}
) -> type:
    r"""Decorator for dataclasses that will be used with Hydra instantiation.

    The decorated dataclass will have a ``_target_`` attribute set with the full
    path of target class ``target``. A method ``to_omegaconf`` will be added to
    facilitate saving with Pytorch Lightning's ``save_hyperparameters``.

    Args:
        target:
            Name of the target class for instantiation. A module path will be
            prepended using the module of the decorated dataclass.
    """
    def caller(clazz, spec=spec):
        if isinstance(clazz, Instantiable):
            if name:
                cs = ConfigStore.instance()
                cs.store(group=group, name=name, node=clazz)
            return clazz

        spec = spec or clazz

        if spec == clazz:
            assert target 
            full_target = f"{clazz.__module__}.{target}"
        else:
            spec_str = f"{spec.__module__}.{spec.__qualname__}"
            full_target = spec_str if target is None else f"{spec_str}.{target}"

        # set _target_ from decorator w/ full module path
        if not hasattr(clazz, "__annotations__"):
            clazz.__annotations__ = {}

        sig = signature(spec)
        if not sig.parameters:
            sig = signature(dataclass(spec))

        any_vkwargs = False
        for k, v in sig.parameters.items():
            # check for *args, *kwargs parameters
            if v.kind == Parameter.VAR_POSITIONAL:
                setattr(clazz, k, field(default_factory=list))
                clazz.__annotations__[k] = List[Any]
                continue
            elif v.kind == Parameter.VAR_KEYWORD:
                setattr(clazz, k, field(default_factory=dict))
                clazz.__annotations__[k] = Dict[str, Any]
                continue

            if k in overrides:
                default = overrides[k]
            else:
                default = v.default if v.default is not v.empty else MISSING

            # Union types not supported by DictConfig, so replace them with Any
            if getattr(v.annotation, "__origin__", None) is Union:
                annotation = Any
            elif v.annotation is v.empty:
                annotation = Any
            else:
                annotation = v.annotation


            # maybe recurse on default
            needs_recurse = not (is_primitive(default) or isinstance(default, Instantiable))

            if needs_recurse and recursive:

                subname = f"{default.__class__.__name__}Conf"
                SubProto = types.new_class(subname)
                SubProto = hydra_dataclass(spec=type(default), recursive=recursive)(SubProto)
                #class SubProto(type(default), metaclass=SubProtoMeta):
                #    ...

                # try to fill in missing values using attributes of default
                subproto = SubProto()
                for f in subproto.__dataclass_fields__.values():
                    if f.default == MISSING and is_primitive(f.type):
                        setattr(subproto, f.name, getattr(default, f.name, MISSING))

                if isinstance(default, torch.utils.data.DataLoader):
                    import pdb; pdb.set_trace()
                default = subproto

            setattr(clazz, k, default)
            clazz.__annotations__[k] = annotation


        def f(self):
            return DictConfig(self)
        clazz.to_omegaconf = f

        clazz._target_ = full_target
        clazz.__annotations__["_target_"] = str
        result = dataclass(clazz)

        if name:
            cs = ConfigStore.instance()
            cs.store(group=group, name=name, node=result)

        return result

    return caller


def is_primitive(x):
    valids = (int, float, bool, str, Enum)
    if x is None: return True
    if isinstance(x, valids): return True
    if isinstance(x, type): return x in valids
    if isinstance(x, (tuple, list)): return all(is_primitive(y) for y in x)
    if isinstance(x, Dict): return all(is_primitive(v) for v in x.values())
    return False


class SubProtoMeta(type):
    def __new__(cls, clsname, superclasses, attributedict):
        sc = superclasses[0]
        clsname = f"{sc.__name__}Conf"
        attributedict["__qualname__"] = f"{sc.__module__}.{clsname}"
        return type.__new__(cls, clsname, tuple(), attributedict)


#def make_dataclass(proto: type, recursive: bool = False) -> type:
#    r"""Decorator used for classes that accept their parameters via a
#    dataclass.
#
#    The decoarated class' ``__init__`` method should accept a single
#    dataclass parameter as protout. A function ``from_args`` will be added
#    that can be used to instantiate the class from raw args by first creating
#    a corresponding dataclass instance, then passing this instance on to ``__init__``.
#
#    Args:
#        spec:
#            Type spec for the dataclass parameter
#    """
#    if not isinstance(proto, type):
#        raise TypeError(f"`proto` must be a type, found {proto}")
#
#    def caller(clazz):
#        if not hasattr(clazz, "__annotations__"):
#            clazz.__annotations__ = {}
#
#        spec = signature(proto)
#        for k, v in spec.parameters.items():
#            default = v.default if v.default is not v.empty else MISSING
#
#            # Union types not supported by DictConfig, so replace them with Any
#            if getattr(v.annotation, "__origin__", None) is Union:
#                annotation = Any
#            elif v.annotation is v.empty:
#                annotation = Any
#            else:
#                annotation = v.annotation
#
#            # maybe recurse on default
#            needs_recurse = not is_primitive(default)
#            if needs_recurse and recursive:
#
#                @make_dataclass(type(default), recursive)
#                class SubProto(type(default), metaclass=SubProtoMeta):
#                    ...
#
#                # try to fill in missing values using attributes of default
#                subproto = SubProto()
#                for f in subproto.__dataclass_fields__.values():
#                    if f.default == MISSING and is_primitive(f.type):
#                        setattr(subproto, f.name, getattr(default, f.name, MISSING))
#
#                default = subproto
#
#            setattr(clazz, k, default)
#            clazz.__annotations__[k] = annotation
#
#        target = f"{proto.__module__}.{proto.__name__}"
#        clazz._target_ = target
#        clazz.__annotations__["_target_"] = str
#
#        return dataclass(clazz)
#
#    return caller


def dataclass_init(spec: type) -> type:
    r"""Decorator used for classes that accept their parameters via a
    dataclass.

    The decoarated class' ``__init__`` method should accept a single
    dataclass parameter as input. A function ``from_args`` will be added
    that can be used to instantiate the class from raw args by first creating
    a corresponding dataclass instance, then passing this instance on to ``__init__``.

    Args:
        spec:
            Type spec for the dataclass parameter
    """
    if not is_dataclass(spec):
        raise TypeError(f"`spec` must be dataclass, found {spec}")

    def caller(clazz):
        def from_args(cls, *args, **kwargs):
            conf = spec(*args, **kwargs)
            return cls(conf)

        setattr(clazz, "from_args", classmethod(from_args))
        return clazz

    return caller


#def hydra_dataclass(target: str, name: Optional[str] = None, group: Optional[str] = None) -> type:
#    r"""Decorator for dataclasses that will be used with Hydra instantiation.
#
#    The decorated dataclass will have a ``_target_`` attribute set with the full
#    path of target class ``target``. A method ``to_omegaconf`` will be added to
#    facilitate saving with Pytorch Lightning's ``save_hyperparameters``.
#
#    Args:
#        target:
#            Name of the target class for instantiation. A module path will be
#            prepended using the module of the decorated dataclass.
#    """
#    if not isinstance(target, str):
#        raise TypeError(f"`target` must be str, found {type(target)}")
#
#    def caller(clazz):
#        # set _target_ from decorator w/ full module path
#        full_target = f"{clazz.__module__}.{target}"
#        clazz._target_ = full_target
#        clazz.__annotations__["_target_"] = str
#
#        def f(self):
#            return DictConfig(self)
#
#        clazz.to_omegaconf = f
#
#        result = dataclass(clazz)
#
#        if name:
#            cs = ConfigStore.instance()
#            cs.store(group=group, name=name, node=result)
#            print(f"Storing name={name}, group={group}")
#
#        return result
#
#    return caller


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
