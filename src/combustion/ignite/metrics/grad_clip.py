#!/usr/bin/env python
# -*- coding: utf-8 -*-
from ignite.contrib.handlers.param_scheduler import LRScheduler
from ignite.engine import Engine, Events
from ignite.metrics import Metric


class GradClipCounter(Metric):
    r"""Tracks learning rate as an Ignite metric as learning rate is updated
    by a LRScheduler.

    Args:
        scheduler (LRScheduler): Scheduler to retrieve current learning rate from
    """

    def __init__(self, scheduler):
        # type: (LRScheduler)
        if not isinstance(scheduler, LRScheduler):
            raise TypeError(f"scheduler must be LRScheduler, got {type(scheduler)}")
        self.scheduler = scheduler.lr_scheduler
        super(ScheduledLR, self).__init__()

    def reset(self):
        pass

    def update(self, output):
        pass

    def compute(self):
        return self.scheduler.get_lr()[0]

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
