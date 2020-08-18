#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import torch

from combustion.nn.functional import fill_normal


class TestFillNormal:
    @pytest.fixture
    def input(self):
        torch.random.manual_seed(42)
        return torch.rand(2, 2, 10, 10)

    @pytest.fixture
    def fill_mask(self):
        torch.random.manual_seed(20)
        return torch.rand(2, 2, 10, 10) > 0.5

    @pytest.fixture
    def sample_mask(self):
        torch.random.manual_seed(10)
        return torch.rand(2, 2, 10, 10) > 0.5

    @pytest.mark.parametrize("preserve_var", [True, False])
    def test_fill(self, input, fill_mask, sample_mask, preserve_var):
        result = fill_normal(input, fill_mask, sample_mask, preserve_var=preserve_var)
        assert result.shape == input.shape
        assert torch.allclose(input[~fill_mask], result[~fill_mask])
        assert (input[fill_mask] != result[fill_mask]).all()
        assert torch.allclose(result.mean(), input.mean(), atol=0.1)
        if preserve_var:
            assert torch.allclose(result.var(), input.var(), atol=0.1)
