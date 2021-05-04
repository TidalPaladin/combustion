#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Optional, Tuple

import torch
from scipy.stats import norm as Normal
from scipy.stats import t as StudentT
from torch import Tensor


class BootstrapMixin:
    r"""Mixin to aid in implementing bootstrapable metrics"""

    @staticmethod
    def gen_bootstraps(total_samples: int, bootstrap_size: int, num_bootstraps: int = 1, seed: int = 42) -> Tensor:
        r"""Generates a set of bootstrap samples.

        Args:
            total_samples (int):
                Total number of samples available to bootstrap from

            bootstrap_size (int):
                Number of samples to select in each bootstrap

            num_bootstraps (int):
                Total number of bootstrap samples to create

            seed (int):
                Seed value for determinisic sampling

        Returns:
            A tensor of indices denoting the samples selected for each bootstrap

        Shape:
            * Output - :math:`(N, L)` where :math:`N` is ``bootstrap_size`` and :math:`L` is  ``num_bootstraps``

        Example:
            >>> # generate 3 bootstrap samples of size 10 from a total sample size of 100
            >>> indices = BootstrapMixin(100, 10, 3)
            >>>
            >>> t = torch.rand(100)
            >>> bootstrap_samples = t[indices] # shape == 10 x 3
        """
        samples: List[Tensor] = []
        with torch.random.fork_rng():
            torch.random.manual_seed(seed)
            for i in range(num_bootstraps):
                samples.append(torch.randperm(total_samples)[:bootstrap_size, None])
        result = torch.cat(samples, dim=-1)
        assert result.shape[0] == bootstrap_size
        assert result.shape[1] == num_bootstraps
        return result

    @staticmethod
    def confidence_interval(
        values: Tensor, ci: float = 0.95, dist="auto", tail="two", unbiased: bool = True, dim=-1
    ) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        r"""Computes a confidence interval for a sample. A Student's T distribution will be used for samples
        smaller than :math:`N=30`. Otherwise a Normal distribution will be used.

        Args:
            values (:class:`torch.Tensor`):
                Sample values

            ci (float):
                The confidence interval compute. Given by :math:`1-\alpha`.

            unbiased (bool):
                Whether to use an unbiased estimator in variance computation

            dist (str):
                Override which distribution to use. Should be ``"auto"``, ``"t"``, or ``"normal"``.

            tail (str):
                Which tailed test to use. Should be ``"left"``, ``"right"``, or ``"two"``.

        Returns:
            Tuple of scalar tensors indicating the lower and upper bounds of the confidence interval.
            For single tailed tests, the non-computed value will be ``None``.
        """
        N = values.shape[dim]
        alpha = 1 - ci

        # compute core statistics for values
        var, mean = torch.var_mean(values, unbiased=unbiased, dim=dim)
        std = var.sqrt()
        se = std / std.new_tensor(N).sqrt_()

        # select distribution
        if dist == "auto":
            dist = "t" if N < 30 else "normal"

        critical_value = BootstrapMixin._get_critical_value(dist, alpha, tail, df=N - 1)
        lower_bound = mean - critical_value * se
        upper_bound = mean + critical_value * se

        if tail == "left":
            return lower_bound, None
        elif tail == "right":
            return None, upper_bound
        elif tail == "two":
            return lower_bound, upper_bound
        else:
            raise ValueError(f"{tail}")

    @staticmethod
    def _get_critical_value(dist: str, alpha: float, tail: str, df: Optional[int] = None) -> float:
        tail = tail.lower()
        dist = dist.lower()

        if dist == "t":

            def crit_func(a):
                return StudentT.ppf(q=a, df=df)

        elif dist == "normal":

            def crit_func(a):
                return Normal.ppf(q=a)

        else:
            raise ValueError(f"{dist}")

        if tail in ("left", "right"):
            q = alpha
        elif tail == "two":
            q = alpha / 2
        else:
            raise ValueError(f"{tail}")

        return abs(crit_func(q))
