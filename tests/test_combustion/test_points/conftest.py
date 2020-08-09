#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest


@pytest.fixture(autouse=True)
def torch_scatter():
    return pytest.importorskip("torch_scatter", reason="test requires torch_scatter")
