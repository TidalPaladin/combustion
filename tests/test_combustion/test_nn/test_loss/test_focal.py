#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import torch
from torch.nn import BCELoss, BCEWithLogitsLoss
from torch.nn.functional import binary_cross_entropy, binary_cross_entropy_with_logits

from combustion.nn import FocalLoss, FocalLossWithLogits, focal_loss, focal_loss_with_logits
from combustion.testing import assert_tensors_close


def test_compare_with_without_logits():
    x1 = torch.rand(10, 10) * 3
    x2 = torch.sigmoid(x1)
    y = torch.rand(10, 10).round()
    with_logits = focal_loss_with_logits(x1, y, gamma=1.0, pos_weight=0.6)
    without_logits = focal_loss(x2, y, gamma=1.0, pos_weight=0.6)
    assert_tensors_close(with_logits, without_logits)


class TestFunctionalFocalLoss:
    @pytest.fixture
    def true_fn(self):
        return binary_cross_entropy

    @pytest.fixture
    def fn(self):
        return focal_loss

    def test_equals_bce_when_gamma_zero(self, fn, true_fn):
        x = torch.rand(10, 10)
        y = torch.rand(10, 10).round()
        true_loss = true_fn(x, y)
        loss = fn(x, y, gamma=0)
        assert_tensors_close(loss, true_loss)

    def test_alpha_when_gamma_zero(self, fn, true_fn):
        x = torch.rand(10, 10)
        y = torch.ones(10, 10)
        loss = fn(x, y, gamma=0, pos_weight=0.0)
        assert loss.item() == 0

    @pytest.mark.parametrize("gamma", [0.0, 0.5, 1.0, 2.0])
    @pytest.mark.parametrize("pos_weight", [None, 0.5, 0.75, 1.0])
    def test_positive_example(self, gamma, pos_weight, fn, true_fn):
        x = torch.Tensor([0.5])
        y = torch.Tensor([1.0])

        bce = true_fn(x, y)
        gamma_term = (1 - x) ** gamma
        true_loss = (pos_weight if pos_weight is not None else 1.0) * gamma_term * bce
        loss = fn(x, y, gamma=gamma, pos_weight=pos_weight)
        assert true_loss.item() == loss.item()

    def test_is_differentiable(self, fn, true_fn):
        x = torch.rand(10, 10, requires_grad=True)
        y = torch.rand(10, 10).round()
        loss = fn(x, y, gamma=0.5, pos_weight=0.5)
        loss.backward()

    def test_label_smoothing(self, fn, true_fn):
        x = torch.rand(10, 10)
        y = torch.rand(10, 10).round()
        smooth_factor = 0.2
        smoothed = y.clamp(smooth_factor, 1 - smooth_factor)
        smoothed_loss = fn(x, y, gamma=0.0, label_smoothing=smooth_factor)
        expected_loss = fn(x, smoothed, gamma=0.0)
        assert_tensors_close(expected_loss, smoothed_loss)

    def test_reduction(self, fn, true_fn):
        x = torch.rand(10, 10)
        y = torch.rand(10, 10).round()
        loss = fn(x, y, gamma=0, reduction="none")
        assert loss.shape == x.shape


class TestFunctionalFocalLossWithLogits(TestFunctionalFocalLoss):
    @pytest.fixture
    def true_fn(self):
        return binary_cross_entropy_with_logits

    @pytest.fixture
    def fn(self):
        return focal_loss_with_logits

    @pytest.mark.parametrize("gamma", [0.0, 0.5, 1.0, 2.0])
    @pytest.mark.parametrize("pos_weight", [None, 0.5, 0.75, 1.0])
    def test_positive_example(self, gamma, pos_weight, fn, true_fn):
        x = torch.Tensor([1.53])
        y = torch.Tensor([1.0])
        p = torch.sigmoid(x)

        bce = true_fn(x, y)
        gamma_term = (1 - p) ** gamma
        true_loss = (pos_weight if pos_weight is not None else 1.0) * gamma_term * bce
        loss = fn(x, y, gamma=gamma, pos_weight=pos_weight)
        assert torch.allclose(true_loss, loss)


class TestFocalLoss:
    @pytest.fixture
    def true_cls(self):
        return BCELoss

    @pytest.fixture
    def cls(self):
        return FocalLoss

    def test_equals_bce_when_gamma_zero(self, cls, true_cls):
        x = torch.rand(10, 10)
        y = torch.rand(10, 10).round()
        true_loss = true_cls()(x, y)
        loss = cls(gamma=0)(x, y)
        assert_tensors_close(loss, true_loss)

    def test_alpha_when_gamma_zero(self, cls, true_cls):
        x = torch.rand(10, 10)
        y = torch.ones(10, 10)
        loss = cls(gamma=0, pos_weight=0.0)(x, y)
        assert loss.item() == 0

    @pytest.mark.parametrize("gamma", [0.0, 0.5, 1.0, 2.0])
    @pytest.mark.parametrize("pos_weight", [None, 0.5, 0.75, 1.0])
    def test_positive_example(self, gamma, pos_weight, cls, true_cls):
        x = torch.Tensor([0.5])
        y = torch.Tensor([1.0])

        bce = true_cls()(x, y)
        gamma_term = (1 - x) ** gamma
        true_loss = (pos_weight if pos_weight is not None else 1.0) * gamma_term * bce
        loss = cls(gamma=gamma, pos_weight=pos_weight)(x, y)
        assert torch.allclose(true_loss, loss)

    def test_is_differentiable(self, cls, true_cls):
        x = torch.rand(10, 10, requires_grad=True)
        y = torch.rand(10, 10).round()
        loss = cls(gamma=0.5, pos_weight=0.5)(x, y)
        loss.backward()

    def test_label_smoothing(self, cls, true_cls):
        x = torch.rand(10, 10)
        y = torch.rand(10, 10).round()
        smooth_factor = 0.2
        smoothed = y.clamp(smooth_factor, 1 - smooth_factor)
        smoothed_loss = cls(gamma=0.0, label_smoothing=smooth_factor)(x, y)
        expected_loss = cls(gamma=0.0)(x, smoothed)
        assert_tensors_close(expected_loss, smoothed_loss)

    def test_reduction(self, cls, true_cls):
        x = torch.rand(10, 10)
        y = torch.rand(10, 10).round()
        loss = cls(gamma=0, reduction="none")(x, y)
        assert loss.shape == x.shape


class TestFocalLossWithLogits(TestFocalLoss):
    @pytest.fixture
    def true_cls(self):
        return BCEWithLogitsLoss

    @pytest.fixture
    def cls(self):
        return FocalLossWithLogits

    @pytest.mark.parametrize("gamma", [0.0, 0.5, 1.0, 2.0])
    @pytest.mark.parametrize("pos_weight", [None, 0.5, 0.75, 1.0])
    def test_positive_example(self, gamma, pos_weight, cls, true_cls):
        x = torch.Tensor([1.53])
        y = torch.Tensor([1.0])
        p = torch.sigmoid(x)

        bce = true_cls()(x, y)
        gamma_term = (1 - p) ** gamma
        true_loss = (pos_weight if pos_weight is not None else 1.0) * gamma_term * bce
        loss = cls(gamma=gamma, pos_weight=pos_weight)(x, y)
        assert torch.allclose(true_loss, loss)
