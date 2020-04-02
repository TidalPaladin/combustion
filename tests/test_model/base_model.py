#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest


@pytest.mark.requires_torch
class BaseModelTest:
    @pytest.fixture
    def module(self):
        raise pytest.UsageError("must override module fixture")

    @pytest.fixture
    def input_shape(self, request):
        raise pytest.UsageError("must override input_shape fixture")

    @pytest.fixture
    def output_shape(self, request):
        raise pytest.UsageError("must override output_shape fixture")

    @pytest.fixture
    def input(self, torch, input_shape):
        return torch.ones(input_shape, requires_grad=True)

    @pytest.fixture
    def target(self, torch, output_shape):
        return torch.ones(output_shape)

    @pytest.fixture(autouse=True)
    def prep_module(self, module):
        module.train()
        module.zero_grad()
        return module

    def test_output_shape(self, module, input, output_shape):
        output = module(input)
        result = tuple(output.shape)
        expected = tuple(output_shape)
        assert result == expected, "expected {}, got {}".format(expected, result)

    def test_all_layers_called(self, torch, module, input, target):
        # run forward / backward pass
        output = module(input)
        loss = torch.nn.BCEWithLogitsLoss()(output, output)
        loss.backward()

        # look for missing gradient in any layer
        missing_grad = []
        for name, param in module.named_parameters():
            if param.grad is None:
                missing_grad.append(name)
        if missing_grad:
            pytest.fail("missing grad for params %s" % missing_grad)
