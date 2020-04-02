#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest


@pytest.fixture(params=["float", "int"], scope="session")
def datagen(request):
    np = pytest.importorskip("numpy", reason="tests require numpy")

    def func(seed, size):
        np.random.seed(seed)
        if request.param == "float":
            return np.random.random_sample(size=size)
        return np.random.randint(0, 254, size=size, dtype=np.int32)

    return func


@pytest.fixture(scope="session")
def plt():
    return pytest.importorskip("matplotlib.pyplot", reason="tests require matplotlib")
