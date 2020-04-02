#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Additional validators for use with Argparse"""
import argparse
import os
import sys
import time


class NumericValidator(argparse.Action):
    """NumericValidator
    Argparse action to validate float or integer values that should
    fall inclusively or exclusively on an interval. 

    Using this action adds the following args to ``add_argument``
        :param low: lower bound on on the interval. default -inf
        :param high: upper bound on the interval. default +inf
        :param inclusive: tuple of bools indicating inclusivity of 
            the lower and upper bound respectively. 
            alternatively, a single bool indicating 
            inclusivity for both lower and upper bounds.

    Example::
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '--foo', type=float, low=0.0, high=1.0, inclusive=(True, False)
        )
    """

    def __init__(self, option_strings, dest, low=None, high=None, inclusive=True, **kwargs):
        super(NumericValidator, self).__init__(option_strings, dest, **kwargs)
        if self.type not in [int, float]:
            raise TypeError("dtype must be one of int, float")
        if low is None and high is None:
            raise ValueError("low and high cannot both be None")
        if low is not None and high is not None and low >= high:
            raise ValueError("must have low < high")

        self.low = self.type(low) if low is not None else float("-inf")
        self.high = self.type(high) if high is not None else float("inf")

        if not isinstance(inclusive, tuple):
            inclusive = (inclusive,) * 2
        if not all([isinstance(x, bool) for x in inclusive]):
            raise TypeError("inclusive must be a single bool or 2-tuple of bools")
        self.inclusive = inclusive

    def __call__(self, parser, namespace, values, option_string=None):
        if isinstance(values, list):
            values = [self._validate(v) for v in values]
        else:
            values = self._validate(values)
        setattr(namespace, self.dest, values)

    def _validate(self, v):
        """Performs validation, raising ArgumentError for invalid inputs"""
        try:
            v = self.type(v)
        except ValueError:
            raise argparse.ArgumentTypeError("could not parse numeric range")

        if not self._check_low(v):
            raise argparse.ArgumentError(self, "value {} exceeded minimum {}".format(v, self.low))
        if not self._check_high(v):
            raise argparse.ArgumentError(self, "value {} exceeded maximum {}".format(v, self.high))
        return v

    def _check_low(self, v):
        """Checks that the lower numeric limit is satisfied"""
        return v >= self.low if self.inclusive[0] else v > self.low

    def _check_high(self, v):
        """Checks that the upper numeric limit is satisfied"""
        return v <= self.high if self.inclusive[0] else v < self.high
