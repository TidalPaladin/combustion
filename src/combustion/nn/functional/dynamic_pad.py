#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Dict, Iterable

import torch.nn as nn

from ..modules import DynamicSamePad


PATCH_TYPES = [
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.MaxPool1d,
    nn.MaxPool2d,
    nn.MaxPool3d,
    nn.AvgPool1d,
    nn.AvgPool2d,
    nn.AvgPool3d,
]


def patch_dynamic_same_pad(
    module: nn.Module,
    padding_mode: str = "constant",
    pad_value: float = 0.0,
    include_classes: Iterable[type] = [],
    include_names: Iterable[str] = [],
    exclude_names: Iterable[str] = [],
) -> Dict[str, nn.Module]:
    r"""Patches spatial layers in a :class:`torch.nn.Module`, wrapping each layer in a
    :class:`combustion.nn.DynamicSamePad` module. This method allows for dynamic same padding
    to be added to a module during or after instantiation.

    .. note::
        This method alone is not sufficient to ensure shape matching throughout a U-Net or similar
        architecture. Use this method in conjunction with :class:`combustion.nn.MatchShapes` for
        correct end to end operation of any input.

    .. warning::
        This method is experimental

    Args:

        module (:class:`torch.nn.Module`):
            The module to patch with dynamic same padding.

        padding_mode (str):
            Padding mode for :class:`combustion.nn.DynamicSamePad`

        pad_value (str):
            Fill value for :class:`combustion.nn.DynamicSamePad`

        include_classes (iterable of types):
            Types of modules to be patched. By default, PyTorch's convolutional and
            pooling layers are matched

        include_names (iterable of str):
            Explicit names of children to be patched. If ``include_names`` is specified,
            only children whose names appear in ``include_names`` will be patched.

        exclude_names (iterable of str):
            Names of children to be excluded from patching.

    Returns:
        A mapping of child module names to their newly patched module instances.
    """
    if not include_classes:
        include_classes = PATCH_TYPES
    kwargs = {
        "padding_mode": padding_mode,
        "pad_value": pad_value,
        "include_classes": include_classes,
        "exclude_names": exclude_names,
    }

    patched: Dict[str, nn.Module] = {}

    # checks if a module is a direct patching target
    def is_patchable(module, module_name):
        if type(module) not in include_classes:
            return False
        if include_names:
            return module_name in include_names
        else:
            return module_name not in exclude_names

    for child_name, child in module.named_children():
        # patch this child if matches a target name/class
        if is_patchable(child, child_name):
            padded = DynamicSamePad(child, padding_mode, pad_value)
            setattr(module, child_name, padded)
            patched[child_name] = padded

        # recurse on patchable subchildren
        patched_children = patch_dynamic_same_pad(child, **kwargs)
        for k, v in patched_children.items():
            patched[f"{child_name}.{k}"] = v

    return patched
