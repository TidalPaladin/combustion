#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gc

import pytest
import torch
from torch import Tensor

from combustion.models import EfficientDet1d, EfficientDet2d, EfficientDet3d
from combustion.nn import MobileNetBlockConfig
from combustion.testing import TorchScriptTestMixin, TorchScriptTraceTestMixin


class EfficientDetBaseTest(TorchScriptTestMixin, TorchScriptTraceTestMixin):
    @pytest.fixture
    def model_type(self):
        raise NotImplementedError()

    @pytest.fixture
    def data(self):
        raise NotImplementedError()

    @pytest.fixture
    def model(self, model_type):
        block1 = MobileNetBlockConfig(4, 8, 3, num_repeats=2, stride=2)
        block2 = MobileNetBlockConfig(8, 16, 3, num_repeats=1, stride=2)
        blocks = [block1, block2]
        model = model_type(blocks, [1, 2, 3, 4])
        yield model
        del model
        gc.collect()

    def test_construct(self, model_type):
        block1 = MobileNetBlockConfig(4, 8, 3, num_repeats=2)
        block2 = MobileNetBlockConfig(8, 16, 3, num_repeats=1)
        blocks = [block1, block2]
        model_type(blocks, [1, 2])
        gc.collect()

    def test_forward(self, model, data):
        output = model(data)
        assert isinstance(output, list)
        assert all([isinstance(x, Tensor) for x in output])

        batch_size, fpn_levels = 1, 64
        data.ndim - 2

        for i, out in enumerate(output):
            assert out.ndim == data.ndim
            assert out.shape[0] == batch_size
            assert out.shape[1] == fpn_levels
            assert tuple(out.shape[2:]) == tuple([x // 2 ** (i + 2) for x in data.shape[2:]])

    def test_backward(self, model, data):
        output = model(data)
        flat = torch.cat([x.flatten() for x in output], dim=-1)
        scalar = flat.sum()
        scalar.backward()

    @pytest.mark.parametrize("compound_coeff", [0, 1, 2])
    def test_from_predefined(self, model_type, compound_coeff, data):
        model = model_type.from_predefined(compound_coeff)
        assert isinstance(model, model_type)

        output = model(data)
        assert isinstance(output, list)
        assert all([isinstance(x, Tensor) for x in output])
        for out in output:
            assert out.ndim == data.ndim
            assert out.shape[0] == 1

        del model


class TestEfficientDet1d(EfficientDetBaseTest):
    @pytest.fixture
    def model_type(self):
        return EfficientDet1d

    @pytest.fixture
    def data(self):
        torch.random.manual_seed(42)
        return torch.rand(1, 3, 256, requires_grad=True)


@pytest.mark.ci_skip
class TestEfficientDet2d(EfficientDetBaseTest):
    @pytest.fixture
    def model_type(self):
        return EfficientDet2d

    @pytest.fixture
    def data(self):
        torch.random.manual_seed(42)
        return torch.rand(1, 3, 256, 256, requires_grad=True)


@pytest.mark.ci_skip
class TestEfficientDet3d(EfficientDetBaseTest):
    @pytest.fixture
    def model_type(self):
        return EfficientDet3d

    @pytest.fixture
    def model(self, model_type):
        block1 = MobileNetBlockConfig(4, 8, (3, 3, 3), num_repeats=2, stride=(1, 2, 2))
        block2 = MobileNetBlockConfig(8, 16, (3, 3, 3), num_repeats=1, stride=(1, 2, 2))
        blocks = [block1, block2]
        model = model_type(blocks, [1, 2, 3, 4], fpn_kwargs={"stride": (1, 2, 2)})
        yield model
        del model
        gc.collect()

    @pytest.fixture
    def data(self):
        torch.random.manual_seed(42)
        return torch.rand(1, 3, 3, 256, 256, requires_grad=True)

    def test_forward(self, model, data):
        output = model(data)
        assert isinstance(output, list)
        assert all([isinstance(x, Tensor) for x in output])

        batch_size, fpn_levels = 1, 64
        data.ndim - 2

        for i, out in enumerate(output):
            assert out.ndim == data.ndim
            assert out.shape[0] == batch_size
            assert out.shape[1] == fpn_levels
            # skip depth dimension check - too slow to manipulate full size 3d tensors
            assert tuple(out.shape[3:]) == tuple([x // 2 ** (i + 2) for x in data.shape[3:]])

    @pytest.mark.parametrize("compound_coeff", [0, 1, 2])
    def test_from_predefined(self, model_type, compound_coeff, data):
        model = model_type.from_predefined(compound_coeff, fpn_kwargs={"stride": (1, 2, 2)})
        assert isinstance(model, model_type)

        output = model(data)
        assert isinstance(output, list)
        assert all([isinstance(x, Tensor) for x in output])
        for out in output:
            assert out.ndim == data.ndim
            assert out.shape[0] == 1

        del model
