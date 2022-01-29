#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from typing import Iterable

import pytest
import torch
from torch import Tensor
from torch.jit import ScriptModule

from combustion.testing import assert_tensors_close, cuda_or_skip


class TorchScriptTestMixin:
    r"""Mixin to test a :class:`torch.nn.Module`'s ability to be scripted using
    :func:`torch.jit.script`, saved to disk, and loaded.

    The following fixtures should be implemented in the subclass:

        * :func:`model` - returns the model to be tested
    """

    @pytest.fixture
    def model(self):
        raise pytest.UsageError(f"Must implement model fixture for {self.__class__.__name__}")

    def test_script(self, model):
        r"""Calls :func:`torch.jit.script` on the given model and tests that a :class:`torch.jit.ScriptModule`
        is returned.
        """
        scripted = torch.jit.script(model)
        assert isinstance(scripted, ScriptModule)

    def test_save_scripted(self, model, tmp_path):
        r"""Calls :func:`torch.jit.script` on the given model and tests that the resultant
        :class:`torch.jit.ScriptModule` can be saved to disk using :func:`torch.jit.save`.
        """
        path = os.path.join(tmp_path, "model.pth")
        scripted = torch.jit.script(model)
        assert isinstance(scripted, ScriptModule)
        torch.jit.save(scripted, path)
        assert os.path.isfile(path)

    def test_load_scripted(self, model, tmp_path):
        r"""Tests that a :class:`torch.jit.ScriptModule` saved to disk using :func:`torch.jit.script` can be
        loaded, and that the loaded object is a :class:`torch.jit.ScriptModule`.
        """
        path = os.path.join(tmp_path, "model.pth")
        scripted = torch.jit.script(model)
        torch.jit.save(scripted, path)
        loaded = torch.jit.load(path)
        assert isinstance(loaded, ScriptModule)


class TorchScriptTraceTestMixin:
    r"""Mixin to test a :class:`torch.nn.Module`'s ability to be traced using
    :func:`torch.jit.trace`, saved to disk, and loaded.

    The following fixtures should be implemented in the subclass:

        * :func:`model` - returns the model to be tested
        * :func:`data` - returns an input to ``model.forward()``.
    """

    @pytest.fixture
    def model(self):
        raise pytest.UsageError(f"Must implement model fixture for {self.__class__.__name__}")

    @pytest.fixture
    def data(self):
        raise pytest.UsageError("Must implement data fixture for {self.__class__.__name__}")

    def test_trace(self, model, data):
        r"""Calls :func:`torch.jit.trace` on the given model and tests that a :class:`torch.jit.ScriptModule`
        is returned.
        """
        traced = torch.jit.trace(model, data)
        assert isinstance(traced, ScriptModule)

    @cuda_or_skip
    def test_traced_forward_call(self, model, data):
        r"""Calls :func:`torch.jit.trace` on the given model and tests that a :class:`torch.jit.ScriptModule`
        is returned.

        Because of the size of some models, this test is only run when a GPU is available.
        """
        traced = torch.jit.trace(model, data)
        output = model(data)
        traced_output = traced(data)
        if isinstance(output, Tensor):
            assert_tensors_close(output, traced_output)
        elif isinstance(output, Iterable):
            for out, traced_out in zip(output, traced_output):
                assert_tensors_close(out, traced_out)
        else:
            pytest.skip()

    def test_save_traced(self, model, tmp_path, data):
        r"""Calls :func:`torch.jit.trace` on the given model and tests that the resultant
        :class:`torch.jit.ScriptModule` can be saved to disk using :func:`torch.jit.save`.
        """
        path = os.path.join(tmp_path, "model.pth")
        traced = torch.jit.trace(model, data)
        assert isinstance(traced, ScriptModule)
        torch.jit.save(traced, path)
        assert os.path.isfile(path)

    def test_load_traced(self, model, tmp_path, data):
        r"""Tests that a :class:`torch.jit.ScriptModule` saved to disk using :func:`torch.jit.trace`
        can be loaded, and that the loaded object is a :class:`torch.jit.ScriptModule`.
        """
        path = os.path.join(tmp_path, "model.pth")
        traced = torch.jit.trace(model, data)
        torch.jit.save(traced, path)
        loaded = torch.jit.load(path)
        assert isinstance(loaded, ScriptModule)
