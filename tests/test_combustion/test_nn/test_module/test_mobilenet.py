#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import torch

from combustion.nn import MobileNetBlockConfig, MobileNetConvBlock1d, MobileNetConvBlock2d, MobileNetConvBlock3d
from combustion.testing import TorchScriptTestMixin, TorchScriptTraceTestMixin


@pytest.mark.filterwarnings("ignore::torch.jit.TracerWarning")
class TestMobileNetConvBlock1d(TorchScriptTestMixin, TorchScriptTraceTestMixin):
    @pytest.fixture
    def model_type(self):
        return MobileNetConvBlock1d

    @pytest.fixture(
        params=[
            pytest.param((3, 1, 1, 0.0, 1, 1, True), id="kernel=3,stride=1,dil=1,dc=0.0,squeeze=1,excite=1,skip=False"),
            pytest.param((5, 2, 1, 0.0, 2, 2, True), id="kernel=3,stride=1,dil=1,dc=0.0,squeeze=2,excite=2,skip=True"),
            pytest.param(
                (3, 1, 1, 0.3, 8, 8, False), id="kernel=3,stride=1,dil=1,dc=0.3,squeeze=8,excite=8,skip=False"
            ),
            pytest.param(
                (3, 1, 2, 0.3, 8, 8, False), id="kernel=3,stride=1,dil=2,dc=0.3,squeeze=8,excite=8,skip=False"
            ),
            pytest.param((5, 1, 4, 0.3, 8, 8, False), id="kernel=5,stride=1,dil=4,dc=0.3,squeeze=8,excite=8,skip=True"),
            pytest.param((5, 1, 4, 0.3, 8, 8, False), id="kernel=5,stride=1,dil=4,dc=0.3,squeeze=8,excite=8,skip=True"),
            pytest.param(
                (5, 1, 4, 0.3, 8, 8, False, False, "avg"),
                id="kernel=5,stride=1,dil=4,dc=0.3,squeeze=8,excite=8,skip=True,global=False,type=avg",
            ),
            pytest.param(
                (5, 1, 4, 0.3, 8, 8, False, True, "max"),
                id="kernel=5,stride=1,dil=4,dc=0.3,squeeze=8,excite=8,skip=True,global=False,type=max",
            ),
        ]
    )
    def model(self, model_type, request):
        kernel = request.param[0]
        stride = request.param[1]
        dil = request.param[2]
        dc = request.param[3]
        squeeze = request.param[4]
        excite = request.param[5]
        skip = request.param[6]

        if len(request.param) > 8:
            request.param[7]
            request.param[8]
        else:
            pass

        return model_type(
            4,
            4,
            kernel,
            stride=stride,
            dilation=dil,
            drop_connect_rate=dc,
            squeeze_excite_ratio=squeeze,
            expand_ratio=excite,
            use_skipconn=skip,
        )

    @pytest.fixture
    def data(self):
        return torch.rand(2, 4, 32)

    def test_construct(self, model_type):
        model_type(4, 4, 3, drop_connect_rate=0.1, squeeze_excite_ratio=2, expand_ratio=2)

    def test_forward(self, model, data):
        output = model(data)
        print(f"Output shape: {output.shape}")
        assert isinstance(output, torch.Tensor)
        assert output.shape[:2] == data.shape[:2]
        if all([x == 1 for x in model._stride]):
            assert output.shape == data.shape
        else:
            assert list(output.shape[2:]) == [x // 2 for x in data.shape[2:]]

    def test_backward(self, model, data):
        output = model(data)
        scalar = output.sum()
        scalar.backward()

    def test_from_config(self, model_type):
        config = MobileNetBlockConfig(
            input_filters=4,
            output_filters=4,
            kernel_size=3,
            stride=1,
            dilation=1,
            drop_connect_rate=0.0,
            squeeze_excite_ratio=2,
            expand_ratio=4,
            use_skipconn=True,
        )
        model = model_type.from_config(config)
        assert isinstance(model, model_type)

    @pytest.mark.parametrize("requires_grad", [True, False])
    @pytest.mark.parametrize("training", [True, False])
    def test_checkpoint(self, model_type, data, requires_grad, training):
        model = model_type(4, 4, 3, drop_connect_rate=0.1, squeeze_excite_ratio=2, expand_ratio=2, checkpoint=True)
        data.requires_grad = requires_grad
        model.train() if training else model.eval()
        output = model(data)

        if requires_grad and training:
            assert "CheckpointFunctionBackward" in output.grad_fn.__class__.__name__
        else:
            assert "CheckpointFunctionBackward" not in output.grad_fn.__class__.__name__


class TestMobileNetConvBlock2d(TestMobileNetConvBlock1d):
    @pytest.fixture
    def model_type(self):
        return MobileNetConvBlock2d

    @pytest.fixture(params=[32, 31])
    def data(self, request):
        return torch.rand(2, 4, *((request.param,) * 2))


class TestMobileNetConvBlock3d(TestMobileNetConvBlock1d):
    @pytest.fixture
    def model_type(self):
        return MobileNetConvBlock3d

    @pytest.fixture
    def data(self):
        return torch.rand(2, 4, 32, 32, 32)


class TestMobileNetBlockConfig:
    @pytest.fixture(params=[1, 2])
    def num_repeats(self, request):
        return request.param

    @pytest.fixture
    def config(self, num_repeats):
        return MobileNetBlockConfig(input_filters=4, output_filters=4, kernel_size=3, num_repeats=num_repeats)

    @pytest.fixture
    def data(self):
        torch.random.manual_seed(42)
        return torch.rand(2, 4, 32, 32)

    def test_get_1d_blocks(self, config, data, num_repeats):
        output = config.get_1d_blocks()
        if num_repeats > 1:
            assert isinstance(output, torch.nn.Sequential)
        else:
            assert isinstance(output, MobileNetConvBlock1d)

    def test_get_2d_blocks(self, config, data, num_repeats):
        output = config.get_2d_blocks()
        if num_repeats > 1:
            assert isinstance(output, torch.nn.Sequential)
        else:
            assert isinstance(output, MobileNetConvBlock2d)

    def test_get_3d_blocks(self, config, data, num_repeats):
        output = config.get_3d_blocks()
        if num_repeats > 1:
            assert isinstance(output, torch.nn.Sequential)
        else:
            assert isinstance(output, MobileNetConvBlock3d)
