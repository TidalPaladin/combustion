#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import torch
import torch.nn.functional as F
from torch.nn import BCELoss, BCEWithLogitsLoss, CrossEntropyLoss
from torch.nn.functional import binary_cross_entropy, binary_cross_entropy_with_logits

from combustion.nn import CategoricalFocalLoss, FocalLoss, FocalLossWithLogits, focal_loss, focal_loss_with_logits
from combustion.testing import assert_tensors_close


@pytest.fixture(autouse=True)
def torch_seed():
    torch.random.manual_seed(42)


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

    @pytest.mark.parametrize("has_positive", [True, False])
    def test_normalize(self, fn, true_fn, has_positive):
        x = torch.rand(10, 10)
        y = torch.rand(10, 10).round()
        if not has_positive:
            y = torch.zeros_like(y)
        num_positive_examples = (y == 1.0).sum().item()

        expected = fn(x, y, 2.0, reduction="none") / max(num_positive_examples, 1)
        norm_loss = fn(x, y, 2.0, reduction="none", normalize=True)
        assert torch.allclose(norm_loss, expected)


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

    def test_stability(self, fn):
        x = torch.Tensor([140, -140])
        y = torch.Tensor([0.0, 1.0])
        loss = fn(x, y, gamma=2.0)
        assert (loss <= 140).all()

    @pytest.mark.usefixtures("cuda_or_skip")
    def test_half(self, fn):
        x1 = torch.tensor([0.0, 140, -140], requires_grad=True).cuda()
        x2 = x1.half()
        y = torch.tensor([1.0, 0.0, 1.0]).cuda()
        loss1 = fn(x1, y, gamma=2.0)
        loss2 = fn(x2, y, gamma=2.0)
        assert torch.allclose(loss1, loss2)

    @pytest.mark.parametrize("scale", [-1000.0, 0.0, 1000.0])
    def test_numerical_stability(self, fn, scale):
        x = torch.rand(1, 5, 10, 10, requires_grad=True)
        y = torch.rand(1, 5, 10, 10)

        loss = fn(x * scale, y, gamma=2.0)
        loss.backward()

        assert not x.grad.isnan().any()


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

    @pytest.mark.parametrize("gamma", [0.0, 2.0])
    @pytest.mark.parametrize("half", [True, False])
    def test_is_differentiable(self, cls, gamma, cuda, half):
        x = torch.rand(10, 10, requires_grad=True)
        y = torch.rand(10, 10).round()

        if cuda:
            x = x.cuda()
            y = y.cuda()

            if half:
                x = x.half()
                y = y.half()

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

    @pytest.mark.parametrize("has_positive", [True, False])
    def test_normalize(self, cls, true_cls, has_positive):
        x = torch.rand(10, 10)
        y = torch.rand(10, 10).round()
        if not has_positive:
            y = torch.zeros_like(y)
        num_positive_examples = (y == 1.0).sum().item()

        expected = cls(2.0, reduction="none")(x, y) / max(num_positive_examples, 1)
        norm_loss = cls(2.0, reduction="none", normalize=True)(x, y)
        assert torch.allclose(norm_loss, expected)


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

    def test_stability(self, cls):
        x = torch.Tensor([140, -140])
        y = torch.Tensor([0.0, 1.0])
        criterion = cls(gamma=2.0)
        loss = criterion(x, y)
        assert (loss <= 140).all()

    @pytest.mark.usefixtures("cuda_or_skip")
    def test_half(self, cls):
        x1 = torch.tensor([0.0, 140, -140], requires_grad=True).cuda()
        x2 = x1.half()
        y = torch.tensor([1.0, 0.0, 1.0]).cuda()
        criterion1 = cls(gamma=2.0)
        criterion2 = cls(gamma=2.0)
        loss1 = criterion1(x1, y)
        loss2 = criterion1(x2, y)
        assert torch.allclose(loss1, loss2)


class TestCategoricalFocalLoss:
    @pytest.fixture
    def true_cls(self):
        kornia = pytest.importorskip("kornia")
        return kornia.losses.FocalLoss

    @pytest.fixture
    def cls(self):
        return CategoricalFocalLoss

    def test_equals_ce_when_gamma_zero(self, cls, true_cls):
        x = torch.rand(1, 5, 10, 10)
        y = torch.randint(0, 5, (1, 10, 10))
        criterion = cls(gamma=0, reduction="none")
        true_criterion = CrossEntropyLoss(reduction="none")
        true_loss = true_criterion(x, y)
        loss = criterion(x, y)
        assert_tensors_close(loss, true_loss)

    @pytest.mark.parametrize("gamma", [0.5, 1.0, 2.0])
    @pytest.mark.parametrize("pos_weight", [None])
    def test_positive_example(self, gamma, cls, true_cls, pos_weight):
        x = torch.rand(1, 5, 10, 10)
        y = torch.randint(0, 5, (1, 10, 10))
        criterion = cls(gamma=gamma, pos_weight=pos_weight, reduction="none")

        p = F.softmax(x, dim=1)
        z = F.one_hot(y, 5).permute(0, -1, 1, 2).contiguous()
        pt = torch.where(z == 1, p, 1 - p)
        pos_weight = torch.as_tensor(pos_weight) if pos_weight is not None else None
        pos_weight = torch.where(z == 1, pos_weight, 1 - pos_weight) if pos_weight is not None else None

        ce = CrossEntropyLoss(reduction="none")(x, y)
        gamma_term = (1 - pt) ** gamma

        true_loss = (pos_weight if pos_weight is not None else 1.0) * gamma_term * ce
        true_loss = true_loss.sum(dim=1).div_(5)
        loss = criterion(x, y)
        assert torch.allclose(true_loss, loss, rtol=0.2)

    @pytest.mark.parametrize("gamma", [0.0, 2.0])
    @pytest.mark.parametrize("half", [True, False])
    def test_is_differentiable(self, cls, gamma, cuda, half):
        x = torch.rand(1, 5, 10, 10, requires_grad=True)
        y = torch.randint(0, 5, (1, 10, 10))

        if cuda:
            x = x.cuda()
            y = y.cuda()
        elif half:
            pytest.skip()

        if half:
            x = x.half()

        criterion = cls(gamma=gamma, reduction="none")
        loss = criterion(x, y).sum()
        loss.backward()

    @pytest.mark.usefixtures("cuda_or_skip")
    def test_half(self, cls):
        x1 = torch.rand(1, 5, 10, 10, requires_grad=True).cuda()
        x2 = x1.half()
        y = torch.randint(0, 5, (1, 10, 10)).cuda()
        criterion1 = cls(gamma=2.0)
        criterion2 = cls(gamma=2.0)
        loss1 = criterion1(x1, y)
        loss2 = criterion1(x2, y)
        assert torch.allclose(loss1, loss2, atol=1e-4)

    @pytest.mark.parametrize("scale", [-1000.0, 0.0, 1000.0])
    def test_numerical_stability(self, cls, scale):
        x = torch.rand(1, 5, 10, 10, requires_grad=True)
        y = torch.randint(0, 5, (1, 10, 10))

        criterion = cls(gamma=2.0)
        loss = criterion(x * scale, y)
        loss.backward()

        assert not x.grad.isnan().any()

    @pytest.mark.parametrize("gamma", [0, 0.5, 1.0])
    @pytest.mark.parametrize("num_classes", [2, 4])
    def test_flat_tensor(self, cls, true_cls, gamma, num_classes):
        x = torch.rand(1, num_classes, 10, 10).view(-1, num_classes)
        y = torch.randint(0, num_classes, (1, 10, 10)).view(-1)
        criterion = cls(gamma=gamma, reduction="none")
        criterion(x, y)

    def test_contiguous(self, cls):
        L, N, C = 100, 2, 3
        x = torch.rand(L, N, C, requires_grad=True)
        y = torch.randint(0, C, (L, N))

        x = x.permute(1, -1, 0)
        y = y.swapdims(0, 1)

        criterion = cls(gamma=2.0)
        loss = criterion(x, y)
        loss.backward()
        assert loss.numel() == 1
