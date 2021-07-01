#!/usr/bin/env python
# -*- coding: utf-8 -*-


import pytest
import torch
from combustion.nn.modules.transformer.perceiver import PerceiverLayer, InitialLatent, PerceiverBlock, PerceiverConfig, PerceiverBlockConfig, Perceiver, PerceiverDualLatent

class TestInitialLatent:

    def test_forward(self):
        Li, N, Di = 512, 2, 64
        Ll, Dl = 32, 64
        x = torch.randn(Li, N, Di)
        l = InitialLatent(Dl, Di, Ll)
        latent = l(x)
        assert latent.shape == (Ll, N, Dl)


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

class TestPerceiverDualLatent:

    def test_forward(self):
        Li, N, Di = 32, 2, 32
        Ll, Dl = 64, 64

        x = torch.randn(Li, N, Di)
        latent = torch.randn(Ll, N, Dl)

        l = PerceiverDualLatent(Dl, Di)
        latent_out, x_out = l(latent, x)
        assert latent_out.shape == (Ll, N, Dl)
        assert x_out.shape == (Li, N, Di)

class TestPerceiverBlock:

    def test_forward(self):
        Li, N, Di = 512, 2, 64
        Ll, Dl = 32, 64

        x = torch.randn(Li, N, Di)

        conf = PerceiverBlockConfig(Dl, Di)
        initializer = conf.initializer(Ll)

        l = PerceiverBlock(conf, initializer, repeats=3)

        latent_out, x_out = l(x)
        assert latent_out.shape == (Ll, N, Dl)
        assert x_out.shape == (Li, N, Di)

class TestPerceiver:

    def test_forward(self):
        Li, N, Di = 512, 2, 32
        x = torch.randn(Li, N, Di)

        conf = PerceiverConfig()
        l = Perceiver.from_config(conf)

        outputs = l(x) 
        assert len(outputs) == len(conf.levels) + 1

    def test_forward_fpn(self):
        Li, N, Di = 512, 2, 32
        x = torch.randn(Li, N, Di)

        conf = PerceiverConfig(fpn_repeats=2)
        l = Perceiver.from_config(conf)

        outputs = l(x) 
        assert len(outputs) == len(conf.levels) + 1
