#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import pytest

from combustion.data.visual import visualize_image, visualize_video


# need this to avoid warning spam
@pytest.fixture(autouse=True)
def clean_pyplot(plt):
    yield None
    plt.close()


class TestVisualizeImage:
    @pytest.fixture
    def png_file(self, tmpdir):
        return os.path.join(tmpdir, "plot.png")

    @pytest.fixture
    def data(self, datagen):
        return datagen(1, (32, 16))

    @pytest.fixture(params=["label", "no_label"])
    def label(self, request, datagen):
        if request.param == "label":
            return datagen(2, (32, 16))
        return None

    def test_returns_figure(self, plt, mocker, data, label, png_file):
        spy = mocker.spy(plt, "subplots")
        actual = visualize_image(data, label, filename=png_file)
        spy.assert_called_once()
        expected = spy.spy_return[0]
        assert actual == expected

    def test_adds_title(self, plt, data, label, png_file):
        title = "my title here"
        fig = visualize_image(data, label, filename=png_file, title=title)
        assert fig._suptitle.get_text() == title

    def test_creates_output_file(self, data, label, png_file):
        visualize_image(data, label, filename=png_file)
        assert os.path.isfile(png_file)

    @pytest.mark.parametrize("label_shape", [pytest.param((1, 32, 16)), pytest.param((32, 16, 1))])
    def test_error_on_ndim_mismatch(self, data, datagen, label_shape, png_file):
        label = datagen(2, label_shape)
        with pytest.raises(ValueError):
            visualize_image(data, label, filename=png_file)

    def test_no_save_file(self, plt, data, label, mocker):
        spy = mocker.spy(plt, "savefig")
        visualize_image(data, label)
        spy.assert_not_called()


class TestVisualizeVideo:
    @pytest.fixture
    def mp4_file(self, tmpdir):
        return os.path.join(tmpdir, "animation.mp4")

    @pytest.fixture
    def data(self, datagen):
        return datagen(1, (20, 32, 16))

    @pytest.fixture(params=["label", "no_label"])
    def label(self, request, datagen):
        if request.param == "label":
            return datagen(2, (20, 32, 16))
        return None

    def test_returns_animation(self, plt, mocker, data, label):
        ani = pytest.importorskip("matplotlib.animation")
        ret = visualize_video(data, 14, label)
        assert isinstance(ret, ani.Animation)

    @pytest.mark.slow
    def test_creates_output_file(self, data, label, mp4_file):
        visualize_video(data, 14, label, mp4_file)
        assert os.path.isfile(mp4_file)

    @pytest.mark.parametrize("label_shape", [pytest.param((31, 16)), pytest.param((32, 15)), pytest.param((32)),])
    def test_error_on_label_shape_mismatch(self, data, datagen, label_shape):
        label = datagen(2, label_shape)
        with pytest.raises(ValueError):
            visualize_video(data, 14, label)

    def test_no_save_file(self, plt, data, label, mocker):
        spy = mocker.spy(plt, "savefig")
        visualize_video(data, 14, label)
        spy.assert_not_called()
