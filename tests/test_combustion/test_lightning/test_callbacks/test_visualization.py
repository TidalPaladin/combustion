#!/usr/bin/env python
# -*- coding: utf-8 -*-


import inspect

import pytest
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from combustion.lightning.callbacks import (
    BlendVisualizeCallback,
    ImageSave,
    KeypointVisualizeCallback,
    VisualizeCallback,
)
from combustion.util import alpha_blend, apply_colormap
from combustion.vision import to_8bit


@pytest.fixture
def trainer(mocker):
    trainer = mocker.MagicMock(spec=pl.Trainer)
    return trainer


def create_image(N, C, H, W):
    torch.random.manual_seed(42)
    return torch.rand(N, C, H, W)


def check_call(call, name, img, step, as_uint8=True):
    if as_uint8:
        img = to_8bit(img, same_on_batch=True)
    assert call.args[0] == name
    assert torch.allclose(call.args[1], img, atol=1)
    assert call.args[2] == step


def assert_calls_equal(call1, call2, **kwargs):
    __tracebackhide__ = True
    call1 = tuple(call1.args) if type(call1) != tuple else call1
    call2 = tuple(call2.args) if type(call2) != tuple else call2
    assert call1[0] == call2[0]
    assert call1[1].shape == call2[1].shape
    assert torch.allclose(call1[1], call2[1], **kwargs)
    assert call1[2] == call2[2]


class TestVisualizeCallback:

    callback_cls = VisualizeCallback

    # training, validation, or testing mode
    @pytest.fixture(params=["train", "val", "test"])
    def mode(self, request):
        return request.param

    # returns a no-args closure of callback function appropriate for `mode`
    @pytest.fixture
    def callback_func(self, mode, trainer, model):
        if mode == "train":
            pass
        elif mode == "val":
            pass
        elif mode == "test":
            pass
        else:
            raise ValueError(f"{mode}")

        def func(self):
            f = getattr(self, func)
            return f(trainer, model)

        return func

    @pytest.fixture
    def data_shape(self):
        return 2, 3, 32, 32

    @pytest.fixture
    def data(self, data_shape):
        data = create_image(*data_shape)
        return data

    @pytest.fixture
    def model(self, request, mocker, callback, data, mode):

        if hasattr(request, "param"):
            step = request.param.pop("step", 10)
            epoch = request.param.pop("epoch", 1)
        else:
            step = 10
            epoch = 1

        model = mocker.MagicMock(name="module")
        model.current_epoch = epoch
        model.global_step = step
        model.global_step = step
        if callback.attr_name is not None:
            setattr(model, callback.attr_name, data)

        if mode == "train":
            attr = "on_train_batch_end"
        elif mode == "val":
            attr = "on_validation_batch_end"
        elif mode == "test":
            attr = "on_test_batch_end"
        else:
            raise ValueError(f"{mode}")
        callback.trigger = lambda: getattr(callback, attr)(trainer, model)

        return model

    @pytest.fixture
    def callback(self, request, trainer):
        cls = self.callback_cls
        init_signature = inspect.signature(cls)
        defaults = {
            k: v.default for k, v in init_signature.parameters.items() if v.default is not inspect.Parameter.empty
        }

        if hasattr(request, "param"):
            name = request.param.get("name", "image")
            defaults.update(request.param)
        else:
            name = "image"
            defaults["name"] = name

        callback = cls(**defaults)
        return callback

    @pytest.fixture
    def logger_func(self, model):
        return model.logger.experiment.add_images

    @pytest.fixture
    def expected_calls(self, data, model, mode, callback):
        if not hasattr(model, callback.attr_name):
            return []

        B, C, H, W = data.shape
        img = [data]
        name = [
            f"{mode}/{callback.name}",
        ]

        # channel splitting
        if callback.split_channels:
            img, name = [], []
            splits = torch.split(data, callback.split_channels, dim=-3)
            for i, s in enumerate(splits):
                n = f"{mode}/{callback.name[i]}"
                name.append(n)
                img.append(s)

        if callback.max_resolution:
            resize_mode = callback.resize_mode
            target = callback.max_resolution
            H_max, W_max = target
            scale_factor = []
            for i in img:
                H, W = i.shape[-2:]
                height_ratio, width_ratio = H / H_max, W / W_max
                s = 1 / max(height_ratio, width_ratio)
                scale_factor.append(s)

            img = [
                F.interpolate(i, scale_factor=s, mode=resize_mode) if s < 1 else i for i, s in zip(img, scale_factor)
            ]

        if (colormap := callback.colormap) :
            if isinstance(colormap, str):
                colormap = [colormap] * len(img)
            img = [apply_colormap(i, cmap)[..., :3, :, :] if cmap is not None else i for cmap, i in zip(colormap, img)]

        if callback.split_batches:
            new_img, new_name = [], []
            for i, n in zip(img, name):
                split_i = torch.split(i, 1, dim=0)
                split_n = [f"{n}/{b}" for b in range(B)]
                new_img += split_i
                new_name += split_n
            name, img = new_name, new_img

        if callback.as_uint8:
            img = [to_8bit(i, same_on_batch=not callback.per_img_norm) for i in img]

        step = [model.current_epoch if callback.epoch_counter else model.global_step] * len(name)
        expected = [(n, i, s) for n, i, s in zip(name, img, step)]
        return expected

    @pytest.mark.parametrize(
        "callback",
        [
            pytest.param(dict(name="image")),
            pytest.param(dict(name="figure")),
            pytest.param(dict(name="image", split_channels=1), marks=pytest.mark.xfail(raises=TypeError)),
            pytest.param(dict(name=["ch0", "ch1", "ch2"], split_channels=1)),
            pytest.param(dict(name=["ch12", "ch3"], split_channels=[2, 1])),
            pytest.param(dict(name="image", split_batches=True)),
            pytest.param(dict(name=["ch0", "ch1", "ch2"], split_channels=1, split_batches=True)),
            pytest.param(dict(name=["ch0", "ch1", "ch2"], split_channels=1, colormap="gnuplot")),
            pytest.param(
                dict(
                    name=["ch0", "ch1", "ch2"],
                    split_channels=1,
                    colormap=[
                        "gnuplot",
                    ]
                    * 3,
                )
            ),
            pytest.param(dict(name=["ch0", "ch1", "ch2"], split_channels=1, colormap=["gnuplot", None, "gnuplot"])),
            pytest.param(dict(name=["ch0", "ch1", "ch2"], split_channels=1, split_batches=True, colormap="gnuplot")),
            pytest.param(dict(name="image", attr_name="other_name")),
            pytest.param(dict(name="image", attr_name=None), marks=pytest.mark.xfail(raises=AttributeError)),
            pytest.param(dict(name="image", attr_name=None, ignore_errors=True)),
            pytest.param(dict(name="image", as_uint8=False)),
            pytest.param(dict(name="image", per_img_norm=True)),
            pytest.param(dict(name="image", as_uint8=False, per_img_norm=True)),
            pytest.param(dict(name="image", epoch_counter=True)),
            pytest.param(dict(name="image", max_resolution=(32, 32))),
            pytest.param(dict(name="image", max_resolution=(16, 16))),
            pytest.param(dict(name="image", max_resolution=(16, 16), resize_mode="nearest")),
        ],
        indirect=True,
    )
    def test_basic_logging(self, model, mode, callback, logger_func, expected_calls):
        callback.trigger()
        if callback.as_uint8:
            atol = 1
        else:
            atol = 0.01

        assert logger_func.call_count == len(expected_calls)
        for actual, expected in zip(logger_func.mock_calls, expected_calls):
            assert_calls_equal(actual, expected, atol=atol)
        assert callback.counter == 1

    @pytest.mark.parametrize(
        "callback",
        [
            pytest.param(dict(name="image", max_calls=10)),
            pytest.param(dict(name=["ch0", "ch1", "ch2"], split_channels=1, max_calls=5)),
            pytest.param(dict(name="image", split_batches=True, max_calls=20)),
            pytest.param(dict(name=["ch0", "ch1", "ch2"], split_channels=1, split_batches=True, max_calls=5)),
        ],
        indirect=True,
    )
    def test_limit_num_images(self, callback, logger_func, data_shape):
        B, C, H, W = data_shape
        num_steps = 20
        for i in range(num_steps):
            callback.trigger()

        limit = callback.max_calls
        expected = min(limit, num_steps) if limit is not None else num_steps

        if callback.split_channels:
            expected *= C
        if callback.split_batches:
            expected *= B

        assert logger_func.call_count == expected

    @pytest.mark.parametrize(
        "model",
        [
            pytest.param(dict(epoch=1, step=1)),
            pytest.param(dict(epoch=1, step=10)),
            pytest.param(dict(epoch=10, step=1)),
            pytest.param(dict(epoch=20, step=20)),
            pytest.param(dict(epoch=32, step=32)),
        ],
        indirect=True,
    )
    @pytest.mark.parametrize(
        "callback",
        [
            pytest.param(dict(name="image", interval=10, epoch_counter=False)),
            pytest.param(dict(name="image", interval=10, epoch_counter=True)),
            pytest.param(dict(name="image", interval=1, epoch_counter=False)),
            pytest.param(dict(name="image", interval=1, epoch_counter=True)),
        ],
        indirect=True,
    )
    def test_log_interval(self, callback, model, logger_func):
        callback.trigger()
        epoch = model.current_epoch
        step = model.global_step
        count_from_epoch = callback.epoch_counter
        interval = callback.interval

        if count_from_epoch and epoch % interval == 0:
            should_log = True
        elif not count_from_epoch and step % interval == 0:
            should_log = True
        else:
            should_log = False

        if should_log:
            logger_func.assert_called()
        else:
            logger_func.assert_not_called()

    @pytest.mark.parametrize(
        "model",
        [
            pytest.param(dict(epoch=1, step=1)),
            pytest.param(dict(epoch=1, step=10)),
            pytest.param(dict(epoch=10, step=1)),
        ],
        indirect=True,
    )
    @pytest.mark.parametrize(
        "callback",
        [
            pytest.param(dict(name="image", epoch_counter=False)),
            pytest.param(dict(name="image", epoch_counter=True)),
        ],
        indirect=True,
    )
    def test_log_custom_fn(self, callback, model, logger_func, tmp_path, mocker):
        PIL = pytest.importorskip("PIL", reason="test requires PIL")
        spy = mocker.spy(PIL.Image.Image, "save")
        callback.log_fn = ImageSave(tmp_path)
        callback.trigger()
        spy.assert_called()
        # TODO make this test more thorough


class TestKeypointVisualizeCallback(TestVisualizeCallback):

    callback_cls = KeypointVisualizeCallback

    @pytest.fixture(params=[pytest.param(True, id="float"), pytest.param(False, id="long")])
    def data(self, data_shape, request):
        B, C, H, W = data_shape
        N = 3
        img = create_image(B, C, H, W)
        bbox = self.create_bbox(B, N)
        bbox = bbox.float() if request.param else bbox.long()
        cls = self.create_classes(B, N)
        score = self.create_classes(B, N)
        target = {"coords": bbox, "class": cls, "score": score}
        return img, target

    def create_bbox(self, B, N):
        torch.random.manual_seed(42)
        bbox = torch.empty(B, N, 4).fill_(-1).float()
        return bbox

    def create_classes(self, B, N):
        torch.random.manual_seed(42)
        return torch.empty(B, N, 1).fill_(-1).float()

    def create_scores(self, B, N):
        torch.random.manual_seed(42)
        return torch.empty(B, N, 1).fill_(-1)

    @pytest.fixture
    def expected_calls(self, data, model, mode, callback, mocker):
        if not hasattr(model, callback.attr_name):
            return []

        data, target = data
        B, C, H, W = data.shape
        img = [data]
        name = [
            f"{mode}/{callback.name}",
        ]

        # channel splitting
        if callback.split_channels:
            img, name = [], []
            splits = torch.split(data, callback.split_channels, dim=-3)
            for i, s in enumerate(splits):
                n = f"{mode}/{callback.name[i]}"
                name.append(n)
                img.append(s)

        if callback.max_resolution:
            resize_mode = callback.resize_mode
            target = callback.max_resolution
            H_max, W_max = target
            needs_resize = [i.shape[-2] > H_max or i.shape[-1] > W_max for i in img]
            img = [F.interpolate(i, target, mode=resize_mode) if resize else i for i, resize in zip(img, needs_resize)]

        if (colormap := callback.colormap) :
            if isinstance(colormap, str):
                colormap = [colormap] * len(img)
            img = [apply_colormap(i, cmap)[..., :3, :, :] if cmap is not None else i for cmap, i in zip(colormap, img)]

        img = [i.repeat(1, 3, 1, 1) if i.shape[-3] == 1 else i for i in img]

        if callback.as_uint8:
            img = [to_8bit(i, same_on_batch=not callback.per_img_norm) for i in img]

        if callback.split_batches:
            new_img, new_name = [], []
            for i, n in zip(img, name):
                split_i = torch.split(i, 1, dim=0)
                split_n = [f"{n}/{b}" for b in range(B)]
                new_img += split_i
                new_name += split_n
            name, img = new_name, new_img

        step = [model.current_epoch if callback.epoch_counter else model.global_step] * len(name)
        expected = [(n, i, s) for n, i, s in zip(name, img, step)]
        return expected


class TestBlendVisualizeCallback(TestVisualizeCallback):

    callback_cls = BlendVisualizeCallback

    @pytest.fixture(params=[pytest.param(True, id="float"), pytest.param(False, id="long")])
    def data(self, data_shape, request):
        B, C, H, W = data_shape
        img = create_image(B, C, H, W)
        return img.clone(), img.clone()

    @pytest.fixture
    def expected_calls(self, data, model, mode, callback, mocker):
        if not hasattr(model, callback.attr_name):
            return []

        if callback.split_channels == (2, 1):
            pytest.skip("incompatible test")

        data1, data2 = data
        B, C, H, W = data1.shape
        img1 = [data1]
        img2 = [data2]
        name1 = [
            f"{mode}/{callback.name}",
        ]
        name2 = [
            f"{mode}/{callback.name}",
        ]
        img = [img1, img2]
        name = [name1, name2]

        # channel splitting

        for pos in range(2):
            if callback.split_channels[pos]:
                img[pos] = []
                name[pos] = []
                img_new, name_new = [], []
                splits = torch.split(data[pos], callback.split_channels[pos], dim=-3)
                for i, s in enumerate(splits):
                    n = f"{mode}/{callback.name[i]}"
                    name_new.append(n)
                    img_new.append(s)
                img[pos] = img_new
                name[pos] = name_new

        if len(img[0]) != len(img[1]):
            if len(img[0]) == 1:
                img[0] = img[0] * len(img[1])
            elif len(img[1]) == 1:
                img[1] = img[1] * len(img[0])
            else:
                raise RuntimeError()

        for pos in range(2):
            if callback.max_resolution:
                resize_mode = callback.resize_mode
                target = callback.max_resolution
                H_max, W_max = target
                needs_resize = [i.shape[-2] > H_max or i.shape[-1] > W_max for i in img[pos]]
                img[pos] = [
                    F.interpolate(i, target, mode=resize_mode) if resize else i
                    for i, resize in zip(img[pos], needs_resize)
                ]

        for pos in range(2):
            if (colormap := callback.colormap[pos]) :
                if isinstance(colormap, str):
                    colormap = [colormap] * len(img[pos])
                img[pos] = [
                    apply_colormap(i, cmap)[..., :3, :, :] if cmap is not None else i
                    for cmap, i in zip(colormap, img[pos])
                ]

        name = name[0]
        final_img = []
        for pos, (d, s) in enumerate(zip(img[0], img[1])):
            B1, C1, H1, W1 = d.shape
            B2, C2, H2, W2 = s.shape

            if C1 != C2:
                if C1 == 1:
                    d = d.repeat(1, C2, 1, 1)
                elif C2 == 1:
                    s = s.repeat(1, C1, 1, 1)
                else:
                    raise ValueError(f"could not match shapes {d.shape}, {s.shape}")

            final_img.append(alpha_blend(d, s, callback.alpha[1], callback.alpha[0])[0])
        img = final_img

        if callback.as_uint8:
            img = [to_8bit(i, same_on_batch=not callback.per_img_norm) for i in img]

        if callback.split_batches:
            new_img, new_name = [], []
            for i, n in zip(img, name):
                split_i = torch.split(i, 1, dim=0)
                split_n = [f"{n}/{b}" for b in range(B)]
                new_img += split_i
                new_name += split_n
            name, img = new_name, new_img

        step = [model.current_epoch if callback.epoch_counter else model.global_step] * len(name)
        expected = [(n, i, s) for n, i, s in zip(name, img, step)]
        return expected

    @pytest.mark.parametrize(
        "callback",
        [
            pytest.param(dict(name="image", max_calls=10)),
            pytest.param(dict(name=["ch0", "ch1", "ch2"], split_channels=1, max_calls=5)),
            pytest.param(dict(name="image", split_batches=True, max_calls=20)),
            pytest.param(dict(name=["ch0", "ch1", "ch2"], split_channels=1, split_batches=True, max_calls=5)),
        ],
        indirect=True,
    )
    def test_limit_num_images(self, callback, logger_func, data_shape):
        B, C, H, W = data_shape
        num_steps = 20
        for i in range(num_steps):
            callback.trigger()

        limit = callback.max_calls
        expected = min(limit, num_steps) if limit is not None else num_steps

        if any(callback.split_channels):
            expected *= C
        if callback.split_batches:
            expected *= B

        assert logger_func.call_count == expected
