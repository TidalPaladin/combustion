#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest


@pytest.fixture(autouse=True, scope="session")
def skip_if_missing():
    pytest.importorskip("torchvision", reason="test requires torchvision")
    pytest.importorskip("kornia", reason="test requires kornia")
    pytest.importorskip("cv2", reason="test requires cv2")
