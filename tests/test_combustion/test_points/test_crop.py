#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import torch

from combustion.points import CenterCrop, center_crop


class TestFunctionalCrop:
    @pytest.mark.parametrize("batched", [True, False])
    @pytest.mark.parametrize("dtype", ["float", "long"])
    def test_crop_big_window_points_unchanged(self, batched, dtype):
        x = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1],])
        if batched:
            x.unsqueeze_(0)
        x = x.long() if dtype == "long" else x.float()
        mask = center_crop(x, (100, 100, 100))
        assert mask.ndim == x.ndim - 1
        assert mask.all()
        assert not batched or mask.shape[0] == 1

    def test_optional_sizes(self):
        x = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1],]).float()
        mask = center_crop(x, (0.5, None, None))
        assert not mask[0]
        assert mask.sum() == 2

    def test_input_unchanged(self):
        x = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1],]).float()
        x_copy = x.clone()
        center_crop(x, (0.5, 0.5, 0.5))
        assert torch.allclose(x, x_copy)

    @pytest.mark.parametrize("batched", [True, False])
    @pytest.mark.parametrize("dtype", ["float", "long"])
    def test_apply_mask_to_coords(self, batched, dtype):
        x = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1],])
        if batched:
            x.unsqueeze_(0).repeat(2, 1, 1)
        x = x.long() if dtype == "long" else x.float()
        mask = center_crop(x, (100, 100, 0.5))
        cropped = x[mask]
        assert torch.allclose(x[..., :2, :], cropped)

    def test_masked_range(self):
        x = torch.tensor([[1, 0, 0], [0, 2, 0], [0, 0, 3],]).float()
        mask = center_crop(x, (2.0, 1.0, 6.1))
        assert mask[0]
        assert not mask[1]
        assert mask[2]


class TestModuleCrop:
    @pytest.mark.parametrize("batched", [True, False])
    @pytest.mark.parametrize("dtype", ["float", "long"])
    def test_crop_big_window_points_unchanged(self, batched, dtype):
        x = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1],])
        if batched:
            x.unsqueeze_(0)
        x = x.long() if dtype == "long" else x.float()
        crop = CenterCrop((100, 100, 100))
        mask = crop(x)
        assert mask.ndim == x.ndim - 1
        assert mask.all()
        assert not batched or mask.shape[0] == 1

    def test_optional_sizes(self):
        x = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1],]).float()
        crop = CenterCrop((0.5, None, None))
        mask = crop(x)
        assert not mask[0]
        assert mask.sum() == 2

    def test_input_unchanged(self):
        x = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1],]).float()
        x_copy = x.clone()
        crop = CenterCrop((0.5, 0.5, 0.5))
        crop(x)
        assert torch.allclose(x, x_copy)

    @pytest.mark.parametrize("batched", [True, False])
    @pytest.mark.parametrize("dtype", ["float", "long"])
    def test_apply_mask_to_coords(self, batched, dtype):
        x = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1],])
        if batched:
            x.unsqueeze_(0).repeat(2, 1, 1)
        x = x.long() if dtype == "long" else x.float()
        crop = CenterCrop((100, 100, 0.5))
        mask = crop(x)
        cropped = x[mask]
        assert torch.allclose(x[..., :2, :], cropped)

    def test_repr(self):
        crop = CenterCrop((0.5, 0.5, 0.5))
        print(crop, end="")
