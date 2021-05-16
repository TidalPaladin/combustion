#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math

import pytest
import pytorch_lightning as pl

from combustion.lightning.mixins import OptimizerMixin


class TestOptimizerMixin:
    @pytest.mark.parametrize(
        "gpus",
        [
            pytest.param(0, id="gpus=0"),
            pytest.param(1, id="gpus=1"),
            pytest.param(2, id="gpus=2"),
        ],
    )
    @pytest.mark.parametrize("accum_grad_batches", [1, 2, 3])
    @pytest.mark.parametrize("num_nodes", [1, 2, 3])
    @pytest.mark.parametrize("steps", [10, 20])
    def test_compute_scheduler_steps(self, steps, gpus, accum_grad_batches, num_nodes):
        actual = OptimizerMixin.compute_scheduler_steps(
            steps,
            gpus,
            num_nodes,
            accum_grad_batches,
        )
        expected = math.ceil(steps / (max(gpus, 1) * num_nodes * accum_grad_batches))
        assert actual == expected

    @pytest.mark.parametrize("dl_len", [None, 10, 20, 30])
    def test_train_steps_per_epoch(self, dl_len):
        mixin = OptimizerMixin()
        if dl_len is not None:
            mixin.train_dataloader = (
                lambda: [
                    None,
                ]
                * dl_len
            )
        assert mixin.train_steps_per_epoch == dl_len

    @pytest.mark.parametrize(
        "max_steps,max_epochs,dl_len,exp",
        [
            pytest.param(100, 3, 100, 100, id="case1"),
            pytest.param(900, 3, 100, 300, id="case2"),
            pytest.param(50, 10, 100, 50, id="case3"),
            pytest.param(None, 10, 100, 1000, id="case4"),
            pytest.param(50, 10, None, 50, id="case5"),
        ],
    )
    def test_train_steps(self, max_steps, max_epochs, dl_len, exp):
        trainer = pl.Trainer(max_steps=max_steps, max_epochs=max_epochs)
        mixin = OptimizerMixin()
        mixin.trainer = trainer
        if dl_len is not None:
            mixin.train_dataloader = (
                lambda: [
                    None,
                ]
                * dl_len
            )
        assert mixin.train_steps == exp
