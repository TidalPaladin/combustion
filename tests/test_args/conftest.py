#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse

import pytest


@pytest.fixture
def mock_args(mocker):
    return mocker.MagicMock(name="args", spec=argparse.Namespace)


@pytest.fixture
def mock_parser(mocker, mock_args):
    m = mocker.MagicMock(name="argparser", spec_set=argparse.ArgumentParser)
    m.parse_args.return_value = mock_args
    mocker.patch("argparse.ArgumentParser", m)
    return m()


@pytest.fixture
def parser(mocker, mock_args):
    parser = argparse.ArgumentParser()
    mocker.spy(parser, "add_argument")
    mocker.spy(parser, "parse_args")
    mocker.patch("argparse.ArgumentParser", return_value=parser)
    return parser
