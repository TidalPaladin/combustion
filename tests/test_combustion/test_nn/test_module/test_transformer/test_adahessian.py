#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from combustion.nn.modules.transformer.adahessian import AdaHessian

class TestAdahessian:

    def test_backward(self):
        model = nn.Sequential(
            nn.Linear(10, 10, bias=True),
            nn.Linear(10, 10, bias=True)
        )
        opt = AdaHessian(model.parameters())
        x = torch.rand(1, 10, 10, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward(create_graph=True)
        opt.step()

    def test_untrainable(self):
        l1 = nn.Linear(10, 10, bias=True)
        l1.bias.trainable = False
        l2 = nn.Linear(10, 10, bias=True)
        l2.bias.trainable = False
        model = nn.Sequential(l1, l2)
        opt = AdaHessian(model.parameters())
        x = torch.rand(1, 10, 10, requires_grad=True)
        out = model(x)
        loss = out.sum()
        opt.zero_grad()
        loss.backward(create_graph=True)
        opt.step()

    def test_unused(self):
        l1 = nn.Linear(10, 10, bias=True)
        l2 = nn.Linear(10, 10, bias=True)
        l3 = nn.Linear(10, 10, bias=True)

        model = nn.ModuleList([l1, l2, l3])
        opt = AdaHessian(model.parameters())
        x = torch.rand(1, 10, 10, requires_grad=True)
        x = model[0](x)
        out = model[1](x)
        loss = out.sum()
        opt.zero_grad()
        loss.backward(create_graph=True)
        opt.step()

        x = model[1](x)
        out = model[2](x)
        loss = out.sum()
        opt.zero_grad()
        loss.backward(create_graph=True)
        opt.step()

    def test_stochastic(self):
        l1 = nn.Linear(10, 10)
        l2 = nn.Linear(10, 10)
        l3 = nn.Linear(10, 20)
        l4 = nn.Linear(20, 20)
        l5 = nn.Linear(20, 20)

        model = nn.ModuleList([l1, l2, l3, l4, l5])
        opt = AdaHessian(model.parameters())

        x = torch.rand(1, 10, 10, requires_grad=True)
        x = model[0](x)
        x = model[1](x)
        x = model[2](x)
        x = model[3](x)
        out = model[4](x)
        loss = out.sum()
        loss.backward(create_graph=True)
        opt.step()
        opt.zero_grad()

        x = torch.rand(1, 10, 10, requires_grad=True)
        x = model[0](x)
        x = model[2](x)
        out = model[4](x)
        loss = out.sum()
        loss.backward(create_graph=True)
        opt.step()
        opt.zero_grad()


    #def test_memory(self):
    #    model = nn.Sequential(
    #        nn.Linear(10, 10, bias=True),
    #        nn.Linear(10, 10, bias=True)
    #    )
    #    opt = AdaHessian(model.parameters())
    #    x = torch.rand(1, 10, 10, requires_grad=True)
    #    for _ in range(1000):
    #        out = model(x)
    #        loss = out.sum()
    #        loss.backward(create_graph=True)
    #        opt.step()
    #    assert False
