#!/usr/bin/env python
# -*- coding: utf-8 -*-


import torch

from combustion.nn.modules.transformer.perceiver import PerceiverLayer


class TestPerceiverLayer:
    def test_forward(self):
        Li, N, Di = 512, 2, 64
        Ll, Dl = 32, 64

        x = torch.randn(Li, N, Di)
        latent = torch.randn(Ll, N, Dl)

        l = PerceiverLayer(Dl, Di)
        latent_out, x_out = l(latent, x)
        assert latent_out.shape == (Ll, N, Dl)
        assert x_out.shape == (Li, N, Di)
