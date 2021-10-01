#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest


@pytest.fixture
def torch_scatter():
    return pytest.importorskip("torch_scatter", reason="test requires torch_scatter")


@pytest.fixture
def torch_cluster():
    return pytest.importorskip("torch_cluster", reason="test requires torch_cluster")
