#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import torch

from combustion.loss import WeightedBCEFromLogitsLoss, WeightedBCELoss, WeightedMSELoss, WeightedSoftMarginLoss


@pytest.fixture(autouse=True, scope="session")
def check_kornia(kornia):
    pass


@pytest.fixture
def rand_input(torch):
    return torch.rand(3, 1, 9, 9)


@pytest.fixture
def rand_target(torch):
    return torch.rand(3, 1, 9, 9)


class _Base:
    _cls = lambda *args, **kwargs: lambda x, y: torch.Tensor(0)

    def test_output_no_sparsity(self, torch):
        input = torch.ones(3, 1, 9, 9)
        target = torch.ones(3, 1, 9, 9)
        criterion = self.__class__._cls(kernel=(3, 3), sigma=(1, 1), sparsity=0)
        loss = criterion(input, target)
        assert loss.item() == 0

    @pytest.mark.parametrize("param", [pytest.param("none"), pytest.param("sum"),])
    def test_reduction(self, param, torch):
        input = torch.zeros(3, 1, 9, 9)
        target = torch.ones(3, 1, 9, 9)
        criterion = self.__class__._cls(kernel=(3, 3), sigma=(1, 1), sparsity=0, reduction=param)
        loss = criterion(input, target)
        if param == "none":
            assert torch.allclose(loss, torch.ones(3, 1, 9, 9))
        else:
            assert loss.item() == torch.ones(3, 1, 9, 9).sum()

    @pytest.mark.parametrize("sparsity", [pytest.param(0), pytest.param(0.001), pytest.param(0.01),])
    def test_sparsity(self, sparsity, torch):
        input = torch.ones(3, 1, 9, 9)
        target = torch.ones(3, 1, 9, 9)
        criterion = self.__class__._cls(kernel=(3, 3), sigma=(1, 1), sparsity=sparsity)
        loss = criterion(input, target)
        expected = sparsity * torch.Tensor([len(input.nonzero())]) / input.numel()
        assert torch.allclose(loss, expected)

    def test_constructor_sigma(self, rand_input, rand_target):
        # best we can do is run with different sigmas and ensure losses are different
        criterion1 = self.__class__._cls(kernel=(3, 3), sigma=(1, 1))
        criterion2 = self.__class__._cls(kernel=(3, 3), sigma=(1.5, 1.5))
        loss1 = criterion1(rand_input, rand_target)
        loss2 = criterion2(rand_input, rand_target)
        assert loss1 != loss2

    def test_constructor_kernel(self, rand_input, rand_target):
        # best we can do is run with different kernels and ensure losses are different
        criterion1 = self.__class__._cls(kernel=(3, 3), sigma=(1, 1))
        criterion2 = self.__class__._cls(kernel=(5, 5), sigma=(1, 1))
        loss1 = criterion1(rand_input, rand_target)
        loss2 = criterion2(rand_input, rand_target)
        assert loss1 != loss2

    def test_getter_sigma(self):
        criterion = self.__class__._cls(kernel=(5, 5), sigma=(2, 2))
        assert criterion.sigma == (2, 2)

    def test_getter_kernel(self):
        criterion = self.__class__._cls(kernel=(5, 5), sigma=(2, 2))
        assert criterion.kernel == (5, 5)

    def test_property_sigma(self, rand_input, rand_target):
        criterion = self.__class__._cls(kernel=(3, 3), sigma=(1, 1))
        loss1 = criterion(rand_input, rand_target)
        criterion.sigma = (1.5, 1.5)
        loss2 = criterion(rand_input, rand_target)
        assert loss1 != loss2

    def test_property_kernel(self, rand_input, rand_target):
        criterion = self.__class__._cls(kernel=(3, 3), sigma=(1, 1))
        loss1 = criterion(rand_input, rand_target)
        criterion.kernel = (5, 5)
        loss2 = criterion(rand_input, rand_target)
        assert loss1 != loss2

    def test_sparsity_scales_with_reduction(self, torch, rand_input, rand_target):
        criterion_none = self.__class__._cls(kernel=(3, 3), sigma=(1, 1), sparsity=0.1, reduction="none")
        criterion_sum = self.__class__._cls(kernel=(3, 3), sigma=(1, 1), sparsity=0.1, reduction="sum")
        criterion_mean = self.__class__._cls(kernel=(3, 3), sigma=(1, 1), sparsity=0.1, reduction="mean")
        loss_none = criterion_none(rand_input, rand_target)
        loss_sum = criterion_sum(rand_input, rand_target)
        loss_mean = criterion_mean(rand_input, rand_target)
        adj_sum = loss_sum / rand_input.numel()
        adj_none = loss_none.sum() / rand_input.numel()
        assert torch.allclose(adj_none, loss_mean)
        assert torch.allclose(adj_sum, loss_mean)


class TestWeightedBCELoss:
    _cls = WeightedBCELoss


class TestWeightedBCEFromLogitsLoss:
    _cls = WeightedBCEFromLogitsLoss


class TestWeightedMSELoss:
    _cls = WeightedMSELoss


class TestWeightedSoftMarginLoss:
    _cls = WeightedSoftMarginLoss
