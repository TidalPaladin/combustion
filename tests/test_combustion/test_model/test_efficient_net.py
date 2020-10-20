#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import torch
from torch import Tensor

from combustion.models import EfficientNet1d, EfficientNet2d, EfficientNet3d
from combustion.nn import MobileNetBlockConfig
from combustion.testing import TorchScriptTestMixin, TorchScriptTraceTestMixin


class EfficientNetBaseTest(TorchScriptTestMixin, TorchScriptTraceTestMixin):
    @pytest.fixture
    def model_type(self):
        raise NotImplementedError()

    @pytest.fixture
    def data(self):
        raise NotImplementedError()

    @pytest.fixture(params=[True, False])
    def model(self, model_type, request):
        checkpoint = request.param
        block1 = MobileNetBlockConfig(4, 8, 3, num_repeats=2, stride=2)
        block2 = MobileNetBlockConfig(8, 16, 3, num_repeats=1, stride=2)
        blocks = [block1, block2]
        return model_type(blocks, 1.0, 1.0, checkpoint=checkpoint)

    @pytest.mark.parametrize("checkpoint", [True, False])
    def test_construct(self, model_type, checkpoint):
        block1 = MobileNetBlockConfig(4, 8, 3, num_repeats=2)
        block2 = MobileNetBlockConfig(8, 16, 3, num_repeats=1)
        blocks = [block1, block2]
        m = model_type(blocks, 1.0, 1.0, checkpoint=checkpoint)
        del m

    def test_forward(self, model, data):
        output = model(data)
        print(f"FPN Shapes: {[x.shape for x in output]}")
        assert isinstance(output, list)
        assert all([isinstance(x, Tensor) for x in output])

        batch_size = 1
        data.ndim - 2

        for i, out in enumerate(output):
            assert out.ndim == data.ndim
            assert out.shape[0] == batch_size
            assert tuple(out.shape[2:]) == tuple([x // 2 ** (i + 2) for x in data.shape[2:]])

    def test_backward(self, model, data):
        output = model(data)
        flat = torch.cat([x.flatten() for x in output], dim=-1)
        scalar = flat.sum()
        scalar.backward()

    @pytest.mark.parametrize("compound_coeff", [0, 1, 2])
    def test_from_predefined(self, model_type, compound_coeff):
        model = model_type.from_predefined(compound_coeff)
        assert isinstance(model, model_type)
        assert model.width_coeff == 1.1 ** compound_coeff
        assert model.depth_coeff == 1.2 ** compound_coeff
        del model

    def test_from_predefined_repeated_calls(self, model_type):
        compound_coeff = 2
        model1 = model_type.from_predefined(compound_coeff)
        model2 = model_type.from_predefined(compound_coeff)
        params1 = sum([x.numel() for x in model1.parameters()])
        params2 = sum([x.numel() for x in model2.parameters()])
        print(f"Params: {params1}")
        assert params1 == params2
        assert params1 > 5e6

    @pytest.mark.parametrize("compound_coeff", [0, 1, 2])
    def test_input_size(self, model_type, compound_coeff, data):
        model = model_type.from_predefined(compound_coeff)
        ndim = data.ndim - 2
        actual1 = model.input_size()
        actual2 = model.input_size(compound_coeff)
        expected = (512 + compound_coeff * 128,) * ndim
        assert actual1 == expected
        assert actual2 == expected
        del model


class TestEfficientNet1d(EfficientNetBaseTest):
    @pytest.fixture
    def model_type(self):
        return EfficientNet1d

    @pytest.fixture
    def data(self):
        torch.random.manual_seed(42)
        return torch.rand(1, 3, 32, requires_grad=True)


@pytest.mark.ci_skip
class TestEfficientNet2d(EfficientNetBaseTest):
    @pytest.fixture
    def model_type(self):
        return EfficientNet2d

    @pytest.fixture
    def data(self):
        torch.random.manual_seed(42)
        return torch.rand(1, 3, 32, 32, requires_grad=True)


@pytest.mark.ci_skip
class TestEfficientNet3d(EfficientNetBaseTest):
    @pytest.fixture
    def model_type(self):
        return EfficientNet3d

    @pytest.fixture
    def data(self):
        torch.random.manual_seed(42)
        return torch.rand(1, 3, 32, 32, 32, requires_grad=True)
