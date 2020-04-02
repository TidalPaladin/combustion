#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from argparse import Namespace

import matplotlib.pyplot as plt
from ignite.contrib.handlers.param_scheduler import ConcatScheduler as _ConcatScheduler
from ignite.contrib.handlers.param_scheduler import CosineAnnealingScheduler as _CosineAnnealingScheduler
from ignite.contrib.handlers.param_scheduler import LRScheduler as _LRScheduler
from ignite.engine import Engine, Events
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ExponentialLR, OneCycleLR


class _AttachMixin:
    def attach(self, engine, event=Events.ITERATION_COMPLETED):
        # type: (Engine, Events.Event) -> LRScheduler
        engine.add_event_handler(event, self)


class LRScheduler(_LRScheduler, _AttachMixin):
    def __init__(self, *args, **kwargs):
        super(LRScheduler, self).__init__(*args, **kwargs)

    @classmethod
    def simulate_from_args(cls, args, scheduler):
        if args.steps is not None:
            steps = args.steps
        else:
            steps = args.epochs * args.steps_per_epoch
        return cls.simulate_values(steps, scheduler.lr_schedule)

    @classmethod
    def visualize_from_args(cls, args, scheduler):
        values = cls.simulate_from_args(args, scheduler)
        if not os.path.isdir(args.result_path):
            os.makedirs(args.result_path)
        path = os.path.join(args.result_path, "lr_schedule.png")
        fig, ax = plt.subplots()
        fig.suptitle(f"{scheduler.__class__.__name__} Schedule")
        ax.set_xlabel("Step")
        ax.set_ylabel("LR")
        ax.ticklabel_format(axis="both", style="sci", scilimits=(-3, 3))
        x, y = list(zip(*values))
        ax.plot(x, y)
        fig.savefig(path)
        plt.close(fig)
        return values

    @classmethod
    def from_args(cls, args, optimizer, last_epoch=-1):
        # type: (Namespace, Optimizer, int)
        if isinstance(args.lr_decay, list) and len(args.lr_decay) > 1:
            return ConcatScheduler.from_args(args, optimizer, last_epoch)
        else:
            args.lr_decay = args.lr_decay[0]

        if args.lr_decay == "one_cycle":
            # pass in steps/epochs/steps_per and let ignite figure out what is not None
            schedule = OneCycleLR(
                optimizer,
                max_lr=args.lr,
                pct_start=args.warmup_percent,
                div_factor=args.div_factor,
                final_div_factor=args.final_div_factor,
                base_momentum=args.lr_base_momentum,
                anneal_strategy=args.lr_anneal,
                max_momentum=args.lr_max_momentum,
                total_steps=args.steps,
                epochs=args.epochs,
                steps_per_epoch=args.steps_per_epoch,
                last_epoch=last_epoch,
            )
            return cls(schedule, save_history=args.lr_save_hist)

        elif args.lr_decay == "exp":
            # pass in steps/epochs/steps_per and let ignite figure out what is not None
            schedule = ExponentialLR(optimizer, gamma=args.lr_decay_coeff, last_epoch=last_epoch,)
            return cls(schedule, save_history=args.lr_save_hist)

        elif args.lr_decay == "cos_ann":
            return CosineAnnealingScheduler.from_args(args, optimizer, last_epoch)

        elif args.lr_decay == "linear":
            return LinearCyclicalScheduler.from_args(args, optimizer, last_epoch)

        elif args.lr_decay is None:
            return None
        else:
            raise ValueError(f"unknown lr decay strategy {args.lr_decay}")
        return cls(schedule)


class _CyclicalScheduler(_AttachMixin):
    @classmethod
    def from_args(cls, args, optimizer, last_epoch=-1):
        scheduler = cls(
            optimizer,
            "lr",
            start_value=args.lr,
            end_value=args.lr / args.div_factor,
            cycle_size=args.lr_cycle_len,
            cycle_mult=args.lr_cycle_mult,
            start_value_mult=args.lr_start_mult,
            end_value_mult=args.lr_end_mult,
        )
        return scheduler

    @classmethod
    def simulate_from_args(cls, args, scheduler):
        if args.steps is not None:
            steps = args.steps
        else:
            steps = args.epochs * args.steps_per_epoch
        return LRScheduler.simulate_values(steps, scheduler)


class CosineAnnealingScheduler(_CosineAnnealingScheduler, _CyclicalScheduler):
    pass


class LinearCyclicalScheduler(_CosineAnnealingScheduler, _CyclicalScheduler):
    pass


class ConcatScheduler(_ConcatScheduler, _AttachMixin):
    pass

    @classmethod
    def from_args(cls, args, optimizer, last_epoch=-1):
        original = args.lr_decay
        schedulers = []
        for x, t in original:
            args.lr_decay = x
            schedulers.append(LRScheduler.from_args(args, optimizer, last_epoch))
        args.lr_decay = original
        return cls(schedulers, t)
