#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest

import combustion.data.preprocessing as preprocessing
from combustion.data.preprocessing import get_class_weights, power_of_two_crop, preprocess, preprocess_frame


@pytest.fixture(autouse=True)
def tv():
    tv = pytest.importorskip("torchvision", reason="test requires torchvision")
    return tv


@pytest.mark.requires_torch
class TestPowerTwoCrop:
    @pytest.fixture(autouse=True)
    def pil(self):
        return pytest.importorskip("PIL", reason="tests require PIL")

    @pytest.fixture(
        params=[
            pytest.param(((33, 17), (32, 16)), id="33x17"),
            pytest.param(((15, 66), (8, 64)), id="8x64"),
            pytest.param(((32, 32), (32, 32)), id="32x32"),
        ]
    )
    def input_shape(self, request):
        return request.param

    @pytest.fixture(params=["tensor", "image"])
    def input(self, request, tv, torch, datagen, input_shape):
        if request.param == "tensor":
            return torch.Tensor(datagen(1, input_shape[0]))
        elif request.param == "image":
            frame = torch.Tensor(datagen(1, input_shape[0]))
            return tv.transforms.functional.to_pil_image(frame)
        raise pytest.UsageError("should have returned already")

    def test_power_of_two_crop(self, torch, tv, datagen, input, input_shape, pil):
        output = power_of_two_crop(input)
        assert isinstance(output, pil.Image.Image), "should return PIL Image"
        assert output.size == input_shape[1]


@pytest.mark.requires_torch
class TestPreprocessFrame:
    @pytest.fixture
    def random(self, mocker):
        random = pytest.importorskip("random")
        m = mocker.MagicMock(spec_set=random.Random())
        mocker.patch.object(random, "Random", return_value=m)
        return m

    @pytest.fixture
    def args(self, mocker):
        args = mocker.MagicMock(name="args")
        args.power_two_crop = False
        args.hflip = False
        args.vflip = False
        args.rotate = 0
        args.brightness = 0.0
        return args

    @pytest.fixture
    def input(self, torch, datagen):
        return torch.Tensor(datagen(1, (1, 32, 32)))

    def test_returns_tensor(self, torch, args, input):
        output = preprocess_frame(input, args, 10, False)
        assert isinstance(output, torch.Tensor)

    @pytest.mark.parametrize("crop", [True, False])
    def test_power_two_crop(self, args, input, mocker, crop):
        spy = mocker.spy(preprocessing, "power_of_two_crop")
        args.power_two_crop = crop
        preprocess_frame(input, args, 10, False)
        if crop:
            spy.assert_called_once()
        else:
            spy.assert_not_called()

    @pytest.mark.parametrize("hflip", [True, False])
    @pytest.mark.parametrize("rand_val", [0.0, 1.0])
    def test_hflip(self, tv, args, input, mocker, hflip, random, rand_val):
        random.random.return_value = rand_val
        spy = mocker.spy(tv.transforms.functional, "hflip")
        args.hflip = hflip
        preprocess_frame(input, args, 10, False)
        if hflip and rand_val == 1.0:
            spy.assert_called_once()
        else:
            spy.assert_not_called()

    @pytest.mark.parametrize("vflip", [True, False])
    @pytest.mark.parametrize("rand_val", [0.0, 1.0])
    def test_vflip(self, tv, args, input, mocker, vflip, random, rand_val):
        random.random.return_value = rand_val
        spy = mocker.spy(tv.transforms.functional, "vflip")
        args.vflip = vflip
        preprocess_frame(input, args, 10, False)
        if vflip and rand_val == 1.0:
            spy.assert_called_once()
        else:
            spy.assert_not_called()

    @pytest.mark.skip(reason="https://github.com/pytorch/vision/issues/1776")
    @pytest.mark.parametrize("rotate", [0, 5, 10])
    def test_rotate(self, tv, args, input, mocker, rotate):
        spy = mocker.spy(tv.transforms.functional, "rotate")
        args.rotate = rotate
        preprocess_frame(input, args, 10, False)
        if rotate > 0:
            spy.assert_called_once()
        else:
            spy.assert_not_called()

    @pytest.mark.parametrize("label", [True, False])
    @pytest.mark.parametrize("brightness", [0.0, 0.05, 0.1])
    def test_brightness(self, tv, args, input, mocker, brightness, label):
        spy = mocker.spy(tv.transforms.functional, "adjust_brightness")
        args.brightness = brightness
        preprocess_frame(input, args, 10, label)
        if not label and brightness:
            spy.assert_called_once()
        else:
            spy.assert_not_called()


@pytest.mark.requires_torch
class TestGetClassWeights:
    @pytest.fixture(
        params=[
            pytest.param(([[1, 0], [0, 1]], [[1, 1], [1, 1]]), id="case1"),
            pytest.param(([[1, 0], [0, 0]], [[1, 1.0 / 3], [1.0 / 3, 1.0 / 3]]), id="case2"),
            pytest.param(([[0, 0], [1, 0]], [[1.0 / 3, 1.0 / 3], [1, 1.0 / 3]]), id="case3"),
        ]
    )
    def data(self, torch, request):
        input = torch.Tensor(request.param[0])
        input = input.unsqueeze(0).unsqueeze(0)
        expected = torch.Tensor(request.param[1])
        expected = expected.unsqueeze(0).unsqueeze(0)
        return input, expected

    def test_get_weights(self, torch, data):
        np = pytest.importorskip("numpy")
        input, expected = data
        output = get_class_weights(input).detach().numpy()
        np.testing.assert_array_equal(output, expected)


class TestPreprocess:
    @pytest.fixture
    def args(self, mocker):
        args = mocker.MagicMock(name="args")
        return args

    @pytest.fixture(autouse=True)
    def preprocess_frame(self, mocker):
        m = mocker.MagicMock(name="args", spec_set=preprocess_frame)
        m.side_effect = lambda *args, **kwargs: args[0]
        mocker.patch("combustion.data.preprocessing.preprocess_frame", m)
        return m

    @pytest.fixture(
        params=[pytest.param((1, 32, 32), id="CxHxW"), pytest.param((4, 1, 32, 32), id="DxCxHxW"),]
    )
    def frames(self, torch, request):
        if isinstance(request.param, list):
            return [torch.ones(x) for x in request.param]
        else:
            return torch.ones(request.param)

    def labels(self, torch, frames):
        return torch.zeros_like(frames)

    @pytest.mark.parametrize(
        "count,shape", [pytest.param(1, (1, 16, 16), id="CxHxW"), pytest.param(2, (1, 16, 16), id="2xCxHxW"),]
    )
    def test_call_frames_and_labels(self, torch, count, shape, args, preprocess_frame):
        frames = torch.ones(count, *shape)
        labels = torch.zeros_like(frames)
        result = preprocess(frames=frames, labels=labels, args=args, seed=42)
        assert preprocess_frame.call_count == 2 * count, "called for every frame + label"
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert torch.allclose(result[0], frames)
        assert torch.allclose(result[1], labels)

    @pytest.mark.parametrize("target", ["frames", "labels"])
    def test_call_frames_or_labels(self, torch, target, args):
        vals = torch.ones(1, 1, 16, 16)
        if target == "frames":
            result = preprocess(frames=vals, args=args, seed=42)
        else:
            result = preprocess(labels=vals, args=args, seed=42)
        assert isinstance(result, torch.Tensor)
        assert torch.allclose(result, vals)

    def test_error_on_frames_and_labels_none(self, args):
        with pytest.raises(ValueError):
            preprocess(args=args, seed=42)
