#!/usr/bin/env python
# -*- coding: utf-8 -*-

import timeit

import torch
from kornia.filters import GaussianBlur2d as KorniaGaussianBlur

from combustion.vision.filters import GaussianBlur2d


def test_fourier_blur():
    torch.random.manual_seed(42)
    inputs = torch.rand(1, 1, 2048, 1024)
    layer = GaussianBlur2d(kernel_size=(91, 91), sigma=(32, 32))
    output = layer(inputs)

    expected = KorniaGaussianBlur((91, 91), (32, 32))(inputs)
    assert torch.allclose(output, expected, atol=0.0001)


def test_fourier_time():
    torch.random.manual_seed(42)
    inputs = torch.rand(1, 1, 1024, 1024)
    layer1 = GaussianBlur2d(kernel_size=(91, 91), sigma=(2, 2))
    layer2 = KorniaGaussianBlur(kernel_size=(91, 91), sigma=(2, 2))

    def func1():
        layer1(inputs)

    def func2():
        layer2(inputs)

    t1 = timeit.timeit(func1, number=2)
    t2 = timeit.timeit(func2, number=2)
    print(f"Gaussian vs Fourier: {t2} vs {t1}")
    assert t1 < t2


def test_repr():
    layer = GaussianBlur2d(kernel_size=(31, 31), sigma=(2, 2))
    print(layer)


def test_grad():
    torch.random.manual_seed(42)
    inputs = torch.rand(1, 1, 2048, 1024, requires_grad=True)
    layer = GaussianBlur2d(kernel_size=(91, 91), sigma=(32, 32))
    output = layer(inputs)
    assert output.grad_fn
