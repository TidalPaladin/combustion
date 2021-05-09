#!/usr/bin/env python
# -*- coding: utf-8 -*-


import pytest
import torch
from combustion.util.masks import edge_dist, get_edges, contract_mask, expand_mask, connect_masks, min_spacing, get_adjacency
from timeit import timeit



class TestEdgeDist:
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

        x = torch.tensor([0, 0, 1, 1, 1, 0, 0])
        l = torch.tensor([-1, -1, 0, 1, 2, -1, -1])
        r = torch.tensor([-1, -1, 2, 1, 0, -1, -1])
        expected = torch.stack((l, r), dim=-2)
        case = pytest.param(x, expected, id='case4')
        cases.append(case)

        return cases

    @pytest.mark.parametrize("mask,expected", get_cases())
    @pytest.mark.parametrize("dtype", ["long", "bool"])
    def test_edge_dist(self, mask, expected, cuda, dtype):
        if dtype == "long":
            mask = mask.long()
        elif dtype == "bool":
            mask = mask.bool()

        if cuda:
            mask = mask.cuda()
            expected = expected.cuda()

        mask = mask.unsqueeze(0)
        expected = expected.unsqueeze(0)
        output = edge_dist(mask)
        for i in range(2 * (output.ndim - 2)):
            output_i =  output[:, i, ...]
            expected_i = expected[:, i, ...]
            assert torch.allclose(output_i, expected_i)

class TestGetEdges:

    def get_cases():
        cases = []
        x = torch.tensor([
           [1, 1, 1, 0, 0],
           [1, 1, 1, 1, 0],
           [0, 1, 1, 0, 0],
           [0, 1, 1, 0, 0],
           [0, 0, 1, 1, 0],
        ])
        edges = torch.tensor([
           [1, 1, 1, 0, 0],
           [1, 1, 1, 1, 0],
           [0, 1, 1, 0, 0],
           [0, 1, 1, 0, 0],
           [0, 0, 1, 1, 0],
        ])
        case = pytest.param(x, edges, True, id='case1')
        cases.append(case)

        x = torch.tensor([
           [1, 1, 1, 1, 0],
           [1, 1, 1, 1, 0],
           [0, 1, 1, 1, 0],
           [0, 1, 1, 0, 0],
           [0, 0, 1, 1, 0],
        ])
        edges = torch.tensor([
           [1, 1, 1, 1, 0],
           [1, 0, 0, 1, 0],
           [0, 1, 0, 1, 0],
           [0, 1, 1, 0, 0],
           [0, 0, 1, 1, 0],
        ])
        case = pytest.param(x, edges, False, id='case2')
        cases.append(case)

        x = torch.tensor([
           [1, 1, 1, 1, 0],
           [1, 1, 1, 1, 0],
           [1, 1, 1, 1, 0],
           [1, 1, 1, 0, 0],
           [0, 0, 1, 1, 0],
        ])
        edges = torch.tensor([
           [1, 1, 1, 1, 0],
           [1, 0, 0, 1, 0],
           [1, 0, 1, 1, 0],
           [1, 1, 1, 0, 0],
           [0, 0, 1, 1, 0],
        ])
        case = pytest.param(x, edges, True, id='case3')
        cases.append(case)

        x = torch.tensor([
           [0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0],
           [1, 1, 1, 0, 0],
        ])
        edges = torch.tensor([
           [0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0],
           [1, 1, 1, 0, 0],
        ])
        case = pytest.param(x, edges, True, id='case4')
        cases.append(case)

        return cases


    @pytest.mark.parametrize("mask,expected,diagonal", get_cases())
    @pytest.mark.parametrize("dtype", ["long", "bool"])
    def test_get_edges(self, mask, expected, cuda, dtype, diagonal):
        if dtype == "long":
            mask = mask.long()
        elif dtype == "bool":
            mask = mask.bool()

        if cuda:
            mask = mask.cuda()
            expected = expected.cuda()

        mask = mask.unsqueeze(0)
        expected = expected.unsqueeze(0).bool().nonzero()
        output = get_edges(mask, diagonal)
        assert (expected == output).all()

class TestContractMask:

    def get_cases():
        cases = []
        x = torch.tensor([
           [0, 0, 1, 0, 0],
           [0, 1, 1, 1, 0],
           [1, 1, 1, 1, 1],
           [0, 1, 1, 1, 0],
           [0, 0, 1, 0, 0],
        ])
        expected = torch.tensor([
           [0, 0, 0, 0, 0],
           [0, 0, 1, 0, 0],
           [0, 1, 1, 1, 0],
           [0, 0, 1, 0, 0],
           [0, 0, 0, 0, 0],
        ])
        case = pytest.param(x, expected, 1, id='case1')
        cases.append(case)

        x = torch.tensor([
           [0, 0, 1, 0, 0],
           [0, 1, 1, 1, 0],
           [1, 1, 1, 1, 1],
           [0, 1, 1, 1, 0],
           [0, 0, 1, 0, 0],
        ])
        expected = torch.tensor([
           [0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0],
           [0, 0, 1, 0, 0],
           [0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0],
        ])
        case = pytest.param(x, expected, 2, id='case2')
        cases.append(case)

        return cases


    @pytest.mark.parametrize("mask,expected, amount", get_cases())
    @pytest.mark.parametrize("dtype", ["long", "bool"])
    def test_contract_mask(self, mask, expected, cuda, dtype, amount):
        if dtype == "long":
            mask = mask.long()
        elif dtype == "bool":
            mask = mask.bool()

        if cuda:
            mask = mask.cuda()
            expected = expected.cuda()

        mask = mask.unsqueeze(0)
        expected = expected.unsqueeze(0).bool()
        output = contract_mask(mask, amount=amount)
        assert (expected == output).all()

class TestExpandMask:

    def get_cases():
        cases = []
        x = torch.tensor([
           [0, 0, 0, 0, 0],
           [0, 0, 1, 0, 0],
           [0, 1, 1, 1, 0],
           [0, 0, 1, 0, 0],
           [0, 0, 0, 0, 0],
        ])
        expected = torch.tensor([
           [0, 1, 1, 1, 0],
           [1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1],
           [0, 1, 1, 1, 0],
        ])
        case = pytest.param(x, expected, 1, id='case1')
        cases.append(case)

        x = torch.tensor([
           [0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0],
           [0, 0, 1, 0, 0],
           [0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0],
        ])
        expected = torch.tensor([
           [1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1],
        ])
        diagonal = True
        case = pytest.param(x, expected, 2, id='case2')
        cases.append(case)

        return cases

    @pytest.mark.parametrize("mask,expected,amount", get_cases())
    @pytest.mark.parametrize("dtype", ["long", "bool"])
    def test_expand_mask(self, mask, expected, cuda, dtype, amount):
        if dtype == "long":
            mask = mask.long()
        elif dtype == "bool":
            mask = mask.bool()

        if cuda:
            mask = mask.cuda()
            expected = expected.cuda()

        mask = mask.unsqueeze(0)
        expected = expected.unsqueeze(0).bool()
        output = expand_mask(mask, amount=amount)
        assert (expected == output).all()

class TestConnectMasks:

    def get_cases():
        cases = []
        x = torch.tensor([
           [1, 1, 0, 0, 0],
           [1, 0, 0, 0, 0],
           [0, 0, 0, 0, 0],
           [0, 0, 0, 0, 1],
           [0, 0, 0, 1, 1],
        ])
        expected = torch.tensor([
           [1, 1, 1, 0, 0],
           [1, 1, 0, 0, 0],
           [1, 0, 0, 0, 1],
           [0, 0, 0, 1, 1],
           [0, 0, 1, 1, 1],
        ])
        case = pytest.param(x, expected, 1, id='case1')
        cases.append(case)

        x = torch.tensor([
           [1, 1, 0, 1, 0],
           [1, 1, 0, 1, 0],
           [0, 1, 0, 0, 0],
           [0, 0, 0, 0, 1],
           [0, 0, 0, 1, 1],
        ])
        expected = x.clone()
        case = pytest.param(x, expected, 1, id='case2')
        cases.append(case)


        x = torch.tensor([
           [1, 1, 0, 0, 0],
           [1, 0, 0, 0, 0],
           [0, 0, 0, 0, 0],
           [0, 0, 0, 0, 1],
           [0, 0, 0, 1, 1],
        ])
        expected = torch.tensor([
           [1, 1, 1, 0, 0],
           [1, 1, 0, 0, 0],
           [1, 0, 0, 0, 1],
           [0, 0, 0, 1, 1],
           [0, 0, 1, 1, 1],
        ])
        case = pytest.param(x, expected, 2, id='case3')
        cases.append(case)

        x = torch.tensor([
           [1, 0, 0, 0, 1],
           [1, 0, 0, 0, 1],
           [1, 0, 0, 0, 1],
           [1, 0, 0, 0, 1],
           [1, 0, 0, 0, 1],
        ])
        expected = torch.tensor([
           [1, 1, 0, 1, 1],
           [1, 1, 0, 1, 1],
           [1, 1, 0, 1, 1],
           [1, 1, 0, 1, 1],
           [1, 1, 0, 1, 1],
        ])
        case = pytest.param(x, expected, 2, id='case4')
        cases.append(case)

        #x = torch.tensor([
        #   [0, 0, 0, 0, 0, 0],
        #   [0, 0, 0, 0, 0, 0],
        #   [0, 0, 1, 1, 0, 0],
        #   [0, 0, 1, 1, 0, 0],
        #   [0, 0, 0, 0, 0, 0],
        #   [0, 0, 0, 0, 0, 0],
        #])
        #expected = x.clone()
        #case = pytest.param(x, expected, 2, id='case5')
        #cases.append(case)

        return cases

    @pytest.mark.parametrize("mask,expected,amount", get_cases())
    @pytest.mark.parametrize("dtype", ["long", "bool"])
    def test_connect_masks(self, mask, expected, cuda, dtype, amount):
        if dtype == "long":
            mask = mask.long()
        elif dtype == "bool":
            mask = mask.bool()

        if cuda:
            mask = mask.cuda()
            expected = expected.cuda()

        mask = mask.unsqueeze(0)
        expected = expected.unsqueeze(0).bool()
        output = connect_masks(mask, 1)
        assert (expected == output).all()

class TestGetAdjacency:

    @pytest.mark.parametrize("self_loops", [True, False])
    @pytest.mark.parametrize("diagonal", [True, False])
    def test_get_adjacency(self, self_loops, diagonal):
        pos = torch.tensor([
            [0, 1, 1],
        ]).unsqueeze_(0)
        N = pos.shape[-2]
        C = pos.shape[-1]

        adjacency = get_adjacency(pos, dims=(-1, -2), diagonal=diagonal, self_loops=self_loops)

        if self_loops and diagonal:
            expected = torch.tensor([
                [0, 0, 0],
                [0, 0, 1],
                [0, 0, 2],
                [0, 1, 0],
                [0, 1, 1],
                [0, 1, 2],
                [0, 2, 0],
                [0, 2, 1],
                [0, 2, 2],
            ])
        elif diagonal:
            expected = torch.tensor([
                [0, 0, 0],
                [0, 0, 1],
                [0, 0, 2],
                [0, 1, 0],
                [0, 1, 2],
                [0, 2, 0],
                [0, 2, 1],
                [0, 2, 2],
            ])
        elif self_loops:
            expected = torch.tensor([
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
                [0, 1, 2],
                [0, 2, 1],
            ])
        else:
            expected = torch.tensor([
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 2],
                [0, 2, 1],
            ])

        adjacency.squeeze_(0)
        assert expected.shape == adjacency.shape

        for coord_e in expected:
            for coord_a in adjacency:
                if (coord_a == coord_e).all():
                    break
            else:
                pytest.fail(f"Missing expected coord: {coord_e}")
