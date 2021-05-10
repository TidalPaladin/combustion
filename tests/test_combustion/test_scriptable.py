#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import torch

from combustion.testing.decorators import scriptable


class TestIsScriptable:
    def test_is_scriptable(self):
        for f in scriptable:
            try:
                torch.jit.script(f)
            except Exception:
                pytest.fail(f"Failed to script {f.__qualname__}")
