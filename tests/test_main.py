#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

import pytest

from combustion.__main__ import main
from combustion.setup import init_logger


class TestInitLogger:
    @pytest.mark.parametrize("dry", [pytest.param(True, id="dry=True"), pytest.param(False, id="dry=False"),])
    def test_init_logger(self, mocker, mock_args, dry):
        mock_args.dry = dry
        mock_args.log_file = "foo"
        spy = mocker.spy(logging, "FileHandler")
        init_logger(mock_args)
        if dry:
            spy.assert_not_called()
        else:
            spy.assert_called()


class TestMain:
    @pytest.fixture(params=[pytest.param("train", id="mode=train"), pytest.param("test", id="mode=test")])
    def mode(self, request, mocker, mock_args):
        mock_args.mode = request.param
        return request.param

    @pytest.fixture
    def parse_args(self, mocker, mock_args, tmp_path):
        mock_args.log_file = "foo"
        mock_args.log_path = tmp_path
        m = mocker.MagicMock(spec_set="combustion.args.parse_args", return_value=mock_args)
        mocker.patch("combustion.args.parse_args", m)
        return m

    @pytest.fixture
    def helpers(self, mocker):
        train = mocker.MagicMock(spec_set="combustion.train.train")
        test = mocker.MagicMock(spec_set="combustion.evaluate.test")
        return train, test

    @pytest.mark.usefixtures("mode")
    def test_parses_args(self, mock_args, parse_args, helpers):
        main(args=["model1", "bce"])
        parse_args.assert_called_once()

    def test_calls_helper(self, mock_args, parse_args, helpers, mode):
        train, test = helpers
        main(args=["model1", "bce"])
        if mode == "train":
            train.assert_called_once()
        else:
            test.assert_called_once()
