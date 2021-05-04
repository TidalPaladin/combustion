#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import torch
from torch import Tensor
from torch.distributions.normal import Normal

from combustion.lightning.metrics import BootstrapMixin


def get_ci_test_cases():
    torch.random.manual_seed(42)
    d_means = (12.4,)
    d_std = (5.1,)
    sample_sizes = (38,)
    use_dist = ("t",)
    cis = (0.99,)
    expected = ((10.2, 14.6),)

    cases = []
    for mean, std, ci, d, size, e in zip(d_means, d_std, cis, use_dist, sample_sizes, expected):
        dist = Normal(mean, std)
        _ = dist.sample(sample_shape=torch.Size([size]))
        case = pytest.param(_, ci, d, e, id=f"{mean}_{std}_{ci}_{d}_{size}")
        cases.append(case)
    return cases


class TestBootstrapMixin:
    @pytest.mark.parametrize(
        "bootstrap_size,samples,num_bootstraps",
        [
            pytest.param(10, 100, 1),
            pytest.param(10, 100, 2),
            pytest.param(50, 100, 2),
            pytest.param(10, 50, 2),
        ],
    )
    def test_gen_bootstraps_output(self, samples, num_bootstraps, bootstrap_size):
        indices = BootstrapMixin.gen_bootstraps(samples, bootstrap_size, num_bootstraps)
        assert indices.shape[0] == bootstrap_size
        assert indices.shape[1] == num_bootstraps
        assert (indices < samples).all()
        assert indices.unique(dim=0).numel() == indices.numel()

    def test_gen_bootstraps_seed(self):
        samples = 100
        num_bootstraps = 100
        bootstrap_size = 50
        indices1 = BootstrapMixin.gen_bootstraps(samples, bootstrap_size, num_bootstraps, seed=42)
        indices2 = BootstrapMixin.gen_bootstraps(samples, bootstrap_size, num_bootstraps, seed=42)
        indices3 = BootstrapMixin.gen_bootstraps(samples, bootstrap_size, num_bootstraps, seed=52)
        assert torch.allclose(indices1, indices2)
        assert not torch.allclose(indices1, indices3)

    @pytest.mark.parametrize(
        "bootstrap_size,samples,num_bootstraps",
        [
            pytest.param(10, 100, 1),
            pytest.param(10, 100, 2),
            pytest.param(50, 100, 2),
            pytest.param(10, 50, 2),
        ],
    )
    def test_gen_bootstraps_index_source(self, samples, num_bootstraps, bootstrap_size):
        torch.random.manual_seed(42)
        t = torch.rand(samples)
        indices = BootstrapMixin.gen_bootstraps(samples, bootstrap_size, num_bootstraps)
        indexed = t[indices]
        assert indexed.shape[0] == bootstrap_size
        assert indexed.shape[1] == num_bootstraps
        assert torch.allclose(indexed[..., 0], t[indices[..., 0]])

    @pytest.mark.parametrize("size", [10, 35])
    @pytest.mark.parametrize("dist", ["t", "normal", "auto"])
    @pytest.mark.parametrize("tail", ["two", "left", "right"])
    def test_confidence_interval_no_error(self, cuda, size, dist, tail):
        torch.random.manual_seed(42)
        sample = torch.rand(size)
        if cuda:
            sample = sample.cuda()
        ci = BootstrapMixin.confidence_interval(sample, 0.95, dist=dist, tail=tail)
        assert isinstance(ci, tuple)
        assert len(ci) == 2
        assert ci[0] is None or isinstance(ci[0], Tensor)
        assert ci[1] is None or isinstance(ci[1], Tensor)

    @pytest.mark.parametrize("sample,ci,dist,expected", get_ci_test_cases())
    def test_confidence_interval_approximate(self, sample, ci, dist, expected):
        ci = BootstrapMixin.confidence_interval(sample, ci, dist=dist)
        expected = tuple(torch.tensor(x) for x in expected)
        assert torch.allclose(expected[0], ci[0], atol=2.0)
        assert torch.allclose(expected[1], ci[1], atol=2.0)

    def test_confidence_interval_exact(self):
        samples = torch.tensor([11, 23.5, 9, 14, 12, 8, 13, 20, 19.1]).view(1, -1).expand(2, -1)
        N = samples.shape[-1]
        var, mean = torch.var_mean(samples, dim=-1)
        se = var.sqrt() / var.new_tensor(N).sqrt_()
        alpha = 0.05
        crit_val = 2.306  # 9 - 1 df
        lb = mean - crit_val * se
        ub = mean + crit_val * se

        ci = BootstrapMixin.confidence_interval(samples, 1 - alpha, dist="t", tail="two")

        assert torch.allclose(ci[0], lb.expand_as(ci[0]))
        assert torch.allclose(ci[1], ub.expand_as(ci[1]))
