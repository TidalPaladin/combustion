#!/usr/bin/env python
# -*- coding: utf-8 -*-
from ignite.contrib.handlers.param_scheduler import LRScheduler
from ignite.engine import Engine, Events
from ignite.metrics import Metric

from ..handlers import CosineAnnealingScheduler


class ScheduledLR(Metric):
    r"""Tracks learning rate as an Ignite metric as learning rate is updated
    by a LRScheduler.

    Args:
        scheduler (LRScheduler): Scheduler to retrieve current learning rate from
    """

    def __init__(self, scheduler):
        # type: (LRScheduler)
        if not isinstance(scheduler, (LRScheduler, CosineAnnealingScheduler)):
            raise TypeError(f"scheduler must be LRScheduler, got {type(scheduler)}")
        if not isinstance(scheduler, CosineAnnealingScheduler):
            self.scheduler = scheduler.lr_scheduler
        else:
            self.scheduler = scheduler
        super(ScheduledLR, self).__init__()
        self.lr = None

    def reset(self):
        pass

    def update(self, output):
        if not isinstance(self.scheduler, CosineAnnealingScheduler):
            self.lr = self.scheduler.get_lr()[0]
        else:
            self.lr = self.scheduler.get_param()

    def compute(self):
        if self.lr is not None:
            return self.lr
        if not isinstance(self.scheduler, CosineAnnealingScheduler):
            return self.scheduler.get_lr()[0]
        else:
            return self.scheduler.get_param()

    def completed(self, engine, name):
        # type: (Engine, str)
        lr = self.compute()
        engine.state.metrics[name] = float(lr)

    def attach(self, engine, name="lr", event_name=Events.ITERATION_COMPLETED):
        # type: (Engine, str, Events.Event)
        r"""
        Attaches the metric to an engine.

        Args:
            engine (Engine): The engine to attach to
        """
        engine.add_event_handler(event_name, self.completed, name)
