#!/usr/bin/env python
# -*- coding: utf-8 -*-


import math
from typing import Optional


def _check_ge_1(var, name):
    if var is not None and var < 1:
        raise ValueError(f"Expected `{name}` >= 1, got {var}")


class OptimizerMixin:
    @property
    def train_steps_per_epoch(self) -> Optional[int]:
        try:
            dl = self.train_dataloader()
            return len(dl)
        except Exception:
            return None

    @property
    def train_steps(self) -> Optional[int]:
        if not hasattr(self, "trainer"):
            return None
        MAX_VAL = 1e12
        trainer = self.trainer
        steps = trainer.max_steps or MAX_VAL
        epoch_steps = (trainer.max_epochs or MAX_VAL) * (self.train_steps_per_epoch or MAX_VAL)
        max_steps = min(steps, epoch_steps)
        return max_steps if max_steps < MAX_VAL else None

    @staticmethod
    def compute_scheduler_steps(steps: int, processors: int = 1, nodes: int = 1, step_interval: int = 1) -> int:
        r"""Computes the total number of steps for a scheduler. This method takes raw inputs and adjusts
        them to account for the effect of multi-GPU, multi-node training and irregular optimizer stepping
        intervals.

        Args:
            steps:
                Raw total number of steps in the loop. Must be provided if
                ``epochs`` and ``steps_per_epoch`` are not provided

            processors:
                Total number of training processes (for multi-GPU training)

            nodes:
                Total number of nodes (for distributed training)

            step_interval:
                Number of batches between each scheduler step
        """
        names = ("steps", "nodes", "step_interval")
        var = (steps, nodes, step_interval)
        for v, n in zip(var, names):
            _check_ge_1(v, n)
        processors = max(processors, 1)

        assert step_interval >= 1
        assert processors >= 1
        assert nodes >= 1

        # steps will be devided across (gpus * num_nodes) workers.
        # steps will be further reduced by factor of accumulate_grad_batches
        divisor = step_interval * processors * nodes

        return math.ceil(steps / divisor)
