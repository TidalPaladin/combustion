#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest


@pytest.fixture(scope="session")
def kornia():
    return pytest.importorskip("kornia")
