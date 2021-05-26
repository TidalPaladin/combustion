#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gc

import pytest
import torch
from torch import Tensor

from combustion.nn import BiFPN, BiFPN1d, BiFPN2d, BiFPN3d
from combustion.testing import TorchScriptTestMixin, cuda_or_skip


class BiFPNBaseTest:
    @pytest.fixture
    def model_class(self):
        raise NotImplementedError()

    @pytest.fixture(params=[1, 4])
    def data(self, request, levels, num_channels):
        raise NotImplementedError()

    @pytest.fixture
    def model(self, model_class, num_channels, levels):
        model = model_class(num_channels, levels)
        yield model
        del model
        gc.collect()

    @pytest.fixture(params=[2, 3])
    def levels(self, request):
        return request.param

    @pytest.fixture(params=[4, 8])
    def num_channels(self, request):
        return request.param

    def test_init(self, levels, num_channels):
        model = BiFPN(num_channels, levels)
        del model
        gc.collect()

    @pytest.mark.parametrize(
        "levels,num_channels,kernel_size,stride,epsilon",
        [
            pytest.param(0, 32, 3, 2, 1e-4, id="levels=0"),
            pytest.param(1, 0, 3, 2, 1e-4, id="num_channels=0"),
            pytest.param(2, 32, 3, 2, 0.0, id="epsilon=0"),
        ],
    )
    def test_validation(self, model_class, levels, num_channels, kernel_size, stride, epsilon):
        with pytest.raises(ValueError):
            model_class(num_channels, levels, kernel_size, stride, epsilon)

    def test_forward(self, model, levels, num_channels, data):
        inputs = data
        # if torch.cuda.is_available():
        #    layer = model.cuda()
        #    output = layer([x.cuda() for x in inputs])
        # else:
        layer = model
        output = layer(inputs)

        assert isinstance(output, list)
        for out_item, in_item in zip(output, inputs):
            assert isinstance(out_item, Tensor)
            assert out_item.shape == in_item.shape
            assert out_item.requires_grad

    def test_backward(self, model, levels, num_channels, data):
        inputs = data
        for x in inputs:
            x.requires_grad = True

        # if torch.cuda.is_available():
        #    layer = model.cuda()
        #    output = layer([x.cuda() for x in inputs])
        # else:
        layer = model
        output = layer(inputs)

        for x in output:
            assert x.requires_grad

        scalar: Tensor = sum([x.sum() for x in output])  # type: ignore
        scalar.backward()

    @pytest.mark.parametrize("training", [True, False])
    @pytest.mark.parametrize("requires_grad", [True, False])
    def test_checkpoint(self, model_class, levels, num_channels, data, training, requires_grad):
        model = model_class(num_channels, levels, checkpoint=True)
        for x in data:
            x.requires_grad = requires_grad
        model.train() if training else model.eval()

        output = model(data)
        for x in output:
            assert not requires_grad or x.requires_grad
            if training and requires_grad:
                assert "CheckpointFunctionBackward" in x.grad_fn.__class__.__name__
            else:
                assert "CheckpointFunctionBackward" not in x.grad_fn.__class__.__name__

        del model
        gc.collect()


class TestBiFPN2d(BiFPNBaseTest):
    @pytest.fixture(params=[BiFPN, BiFPN2d])
    def model_class(self, request):
        return request.param

    @pytest.fixture
    def data(self, levels, num_channels):
        batch_size = 1
        base_size = 8
        result = []
        for i in range(levels):
            height, width = (base_size * 2 ** i,) * 2
            t = torch.rand(batch_size, num_channels, height, width)
            result.append(t)
        return list(reversed(result))

    @pytest.mark.skip
    @cuda_or_skip
    def test_known_input(self, model_class):
        torch.random.manual_seed(42)
        layer = model_class(2, 3)
        input = [
            torch.rand(1, 2, 16, 16),
            torch.rand(1, 2, 8, 8),
            torch.rand(1, 2, 4, 4),
        ]
        output = layer(input)

        # outout maps are large, so just compare the sums of elements in each level
        output_sum = [x.sum() for x in output]
        expected_output_sum = [torch.tensor(205.5710469), torch.tensor(51.3697109), torch.tensor(12.5726123)]

        for level, (expected, actual) in enumerate(zip(expected_output_sum, output_sum)):
            assert torch.allclose(expected, actual, rtol=0.1)

    @cuda_or_skip
    def test_shape_matching(self, model_class):
        torch.random.manual_seed(42)
        layer = model_class(2, 3)
        input = [
            torch.rand(1, 2, 15, 15),
            torch.rand(1, 2, 7, 7),
            torch.rand(1, 2, 3, 3),
        ]
        output = layer(input)
        for x, y in zip(input, output):
            assert x.shape[2:] == y.shape[2:]


class TestBiFPN1d(BiFPNBaseTest):
    @pytest.fixture
    def model_class(self):
        return BiFPN1d

    @pytest.fixture
    def data(self, levels, num_channels):
        batch_size = 1
        base_size = 2
        result = []
        for i in range(levels):
            height, width = (base_size * 2 ** i,) * 2
            t = torch.rand(batch_size, num_channels, height)
            result.append(t)
        return list(reversed(result))

    @pytest.mark.skip
    @cuda_or_skip
    def test_known_input(self, model_class):
        torch.random.manual_seed(42)
        layer = model_class(2, 3)
        input = [
            torch.rand(1, 2, 16),
            torch.rand(1, 2, 8),
            torch.rand(1, 2, 4),
        ]
        output = layer(input)

        # outout maps are large, so just compare the sums of elements in each level
        output_sum = [x.sum() for x in output]

        expected_output_sum = [torch.tensor(12.7858945), torch.tensor(6.7286519), torch.tensor(3.4909187)]

        for level, (expected, actual) in enumerate(zip(expected_output_sum, output_sum)):
            assert torch.allclose(expected, actual, rtol=0.1)

    @cuda_or_skip
    def test_shape_matching(self, model_class):
        torch.random.manual_seed(42)
        layer = model_class(2, 3)
        input = [
            torch.rand(1, 2, 15),
            torch.rand(1, 2, 7),
            torch.rand(1, 2, 3),
        ]
        output = layer(input)
        for x, y in zip(input, output):
            assert x.shape[2:] == y.shape[2:]


class TestBiFPN3d(BiFPNBaseTest):
    @pytest.fixture
    def model_class(self):
        return BiFPN3d

    @pytest.fixture
    def data(self, levels, num_channels):
        batch_size = 1
        base_size = 8
        result = []
        for i in range(levels):
            depth, height, width = (base_size * 2 ** i,) * 3
            t = torch.rand(batch_size, num_channels, depth, height, width)
            result.append(t)
        return list(reversed(result))

    @pytest.mark.skip
    @cuda_or_skip
    def test_known_input(self, model_class):
        torch.random.manual_seed(42)
        layer = model_class(2, 3)
        input = [
            torch.rand(1, 2, 16, 16, 16),
            torch.rand(1, 2, 8, 8, 8),
            torch.rand(1, 2, 4, 4, 4),
        ]
        output = layer(input)

        # outout maps are large, so just compare the sums of elements in each level
        output_sum = [x.sum() for x in output]

        expected_output_sum = [torch.tensor(3230.172), torch.tensor(388.793), torch.tensor(52.6986758)]

        for level, (expected, actual) in enumerate(zip(expected_output_sum, output_sum)):
            assert torch.allclose(expected, actual, rtol=0.1)


class BiFPNScriptBaseTest(TorchScriptTestMixin):
    @pytest.fixture
    def model_class(self):
        raise NotImplementedError()

    @pytest.fixture
    def data(self):
        raise NotImplementedError()

    @pytest.fixture
    def model(self, model_class):
        yield model_class(8, 2)
        gc.collect()


class TestBiFPN2dScript(BiFPNScriptBaseTest):
    @pytest.fixture
    def model_class(self):
        return BiFPN2d

    @pytest.fixture
    def data(self):
        batch_size = 1
        base_size = 8
        levels = 2
        num_channels = 8
        result = []
        for i in range(levels):
            height, width = (base_size * 2 ** i,) * 2
            t = torch.rand(batch_size, num_channels, height, width)
            result.append(t)
        return list(reversed(result))


class TestBiFPN1dScript(BiFPNScriptBaseTest):
    @pytest.fixture
    def model_class(self):
        return BiFPN1d

    @pytest.fixture
    def data(self):
        batch_size = 1
        base_size = 8
        levels = 2
        num_channels = 8
        result = []
        for i in range(levels):
            height, width = (base_size * 2 ** i,) * 2
            t = torch.rand(batch_size, num_channels, height)
            result.append(t)
        return list(reversed(result))


class TestBiFPN3dScript(BiFPNScriptBaseTest):
    @pytest.fixture
    def model_class(self):
        return BiFPN3d

    @pytest.fixture
    def data(self):
        batch_size = 1
        base_size = 8
        levels = 2
        num_channels = 8
        result = []
        for i in range(levels):
            height, width, depth = (base_size * 2 ** i,) * 3
            t = torch.rand(batch_size, num_channels, depth, height, width)
            result.append(t)
        return list(reversed(result))

    @cuda_or_skip
    def test_shape_matching(self, model_class):
        torch.random.manual_seed(42)
        layer = model_class(2, 3)
        input = [
            torch.rand(1, 2, 15, 15, 15),
            torch.rand(1, 2, 7, 7, 7),
            torch.rand(1, 2, 3, 3, 3),
        ]
        output = layer(input)
        for x, y in zip(input, output):
            assert x.shape[2:] == y.shape[2:]
