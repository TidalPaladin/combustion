#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

from combustion.args import parse_args


def test_parse_defaults():
    input = ["model1", "bce"]
    args = parse_args(args=input)
    assert isinstance(args, argparse.Namespace)
