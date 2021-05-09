#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import torch
from combustion.nn import PolygonLoss
from timeit import timeit

def get_cases():
    cases = []
    x = torch.tensor([
       [1, 1, 1, 1],
       [1, 1, 1, 1],
       [1, 1, 1, 1],
       [1, 1, 1, 1],
    ])
    l = torch.tensor([
       [0, 1, 2, 3],
       [0, 1, 2, 3],
       [0, 1, 2, 3],
       [0, 1, 2, 3],
    ])
    t = l.T
    r = l.fliplr()
    b = t.flipud()
    expected = torch.stack((l, t, r, b), dim=-3)
    case = pytest.param(x, expected, id='case1')
    cases.append(case)


    x = torch.tensor([
       [1, 1, 1, 0, 0],
       [1, 1, 1, 1, 0],
       [0, 1, 1, 0, 0],
       [0, 1, 1, 0, 0],
       [0, 0, 1, 1, 0],
    ])
    l = torch.tensor([
       [0, 1, 2, -1, -1],
       [0, 1, 2, 3, -1],
       [-1, 0, 1, -1, -1],
       [-1, 0, 1, -1, -1],
       [-1, -1, 0, 1, -1],
    ])
    t = torch.tensor([
       [0, 0, 0, -1, -1],
       [1, 1, 1, 0, -1],
       [-1, 2, 2, -1, -1],
       [-1, 3, 3, -1, -1],
       [-1, -1, 4, 0, -1],
    ])
    r = torch.tensor([
       [2, 1, 0, -1, -1],
       [3, 2, 1, 0, -1],
       [-1, 1, 0, -1, -1],
       [-1, 1, 0, -1, -1],
       [-1, -1, 1, 0, -1],
    ])
    b = torch.tensor([
       [1, 3, 4, -1, -1],
       [0, 2, 3, 0, -1],
       [-1, 1, 2, -1, -1],
       [-1, 0, 1, -1, -1],
       [-1, -1, 0, 0, -1],
    ])
    expected = torch.stack((l, t, r, b), dim=-3)
    case = pytest.param(x, expected, id='case2')
    cases.append(case)


    x = torch.tensor([
       [1, 1, 0, 1, 1, 1],
       [1, 1, 0, 1, 1, 1],
       [1, 1, 0, 1, 1, 1],
       [1, 1, 0, 1, 1, 1],
       [0, 0, 0, 1, 1, 1],
    ])
    l = torch.tensor([
       [0, 1, -1, 0, 1, 2],
       [0, 1, -1, 0, 1, 2],
       [0, 1, -1, 0, 1, 2],
       [0, 1, -1, 0, 1, 2],
       [-1, -1, -1, 0, 1, 2],
    ])
    t = torch.tensor([
       [0, 0, -1, 0, 0, 0],
       [1, 1, -1, 1, 1, 1],
       [2, 2, -1, 2, 2, 2],
       [3, 3, -1, 3, 3, 3],
       [-1, -1, -1, 4, 4, 4],
    ])
    r = torch.tensor([
       [1, 0, -1, 2, 1, 0],
       [1, 0, -1, 2, 1, 0],
       [1, 0, -1, 2, 1, 0],
       [1, 0, -1, 2, 1, 0],
       [-1, -1, -1, 2, 1, 0],
    ])
    b = torch.tensor([
       [3, 3, -1, 4, 4, 4],
       [2, 2, -1, 3, 3, 3],
       [1, 1, -1, 2, 2, 2],
       [0, 0, -1, 1, 1, 1],
       [-1, -1, -1, 0, 0, 0],
    ])
    expected = torch.stack((l, t, r, b), dim=-3)
    case = pytest.param(x, expected, id='case3')
    cases.append(case)
    return cases

@pytest.mark.parametrize("mask,expected", get_cases())
@pytest.mark.parametrize("dtype", ["long", "bool"])
def test_polygon_loss(mask, expected, cuda, dtype):
    if dtype == "long":
        mask = mask.long()
    elif dtype == "bool":
        mask = mask.bool()

    if cuda:
        mask = mask.cuda()
        expected = expected.cuda()

    output = PolygonLoss.create_targets(mask)
    assert torch.allclose(output, expected)

def test_polygon_loss_runtime():
    x = torch.ones(512, 512)
    def func():
        PolygonLoss.create_targets(x)

    t = timeit(func, number=5) / 5
    print(f"PolygonLoss.create_targets: {t}s")

def test_polygon_postprocess():
    x = torch.ones(5, 8)
    H, W = x.shape[-2:]
    _ = PolygonLoss.create_targets(x).view(1, 4, H, W)
    output = PolygonLoss.postprocess(_, x.bool().view(1, H, W))
    assert output.sum() == x.sum() * 4
    assert (output[..., 0, 0] == 13).all()
    assert (output[..., 0, -1] == 13).all()
    assert (output[..., -1, 0] == 13).all()
    assert (output[..., -1, -1] == 13).all()
    assert (output[..., 0, 1:-1] == 5).all()
    assert (output[..., -1, 1:-1] == 5).all()
    assert (output[..., 1:-1, 0] == 8).all()
    assert (output[..., 1:-1, -1] == 8).all()
    assert (output[..., 1:-1, 1:-1] == 0).all()

def test_polygon_postprocess2():
    x = torch.tensor([
       [0, 0, 1, 0, 0],
       [0, 1, 1, 1, 0],
       [1, 1, 1, 1, 1],
       [0, 1, 1, 1, 0],
       [0, 0, 1, 0, 0],
    ])
    H, W = x.shape[-2:]

    expected = torch.tensor([
        [0, 0, 7, 0, 0],
        [0, 6, 0, 6, 0],
        [7, 0, 0, 0, 7],
        [0, 6, 0, 6, 0],
        [0, 0, 7, 0, 0]
    ]).view(1, H, W)

    _ = PolygonLoss.create_targets(x).view(1, 4, H, W)
    output = PolygonLoss.postprocess(_, x.bool().view(1, H, W))
    assert output.sum() == x.sum() * 4
    assert torch.allclose(output, expected)

def test_polygon_postprocess3():
    x = torch.tensor([
       [0, 0, 1, 0, 0, 0, 0, 0, 0],
       [0, 1, 1, 1, 1, 0, 1, 1, 0],
       [1, 1, 1, 1, 1, 1, 1, 1, 1],
       [0, 1, 1, 1, 1, 1, 0, 0, 0],
       [0, 0, 1, 1, 1, 0, 0, 0, 0],
    ])
    H, W = x.shape[-2:]

    expected = torch.tensor([
        [ 0,  0,  7,  0,  0,  0,  0,  0,  0],
        [ 0,  7,  0,  4,  8,  0,  4,  4,  0],
        [11,  0,  0,  0,  0,  2,  2,  2, 11],
        [ 0,  8,  0,  0,  0,  7,  0,  0,  0],
        [ 0,  0,  8,  4,  7,  0,  0,  0,  0]
    ]).view(1, H, W)

    _ = PolygonLoss.create_targets(x).view(1, 4, H, W)
    output = PolygonLoss.postprocess(_, x.bool().view(1, H, W))
    assert torch.allclose(expected, output)
