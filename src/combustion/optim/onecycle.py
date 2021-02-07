#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional

from torch.optim import Optimizer
from torch.optim.lr_scheduler import OneCycleLR


class SuperConvergenceLR(OneCycleLR):
    r"""Sets the learning rate of each parameter group according to the
    1cycle learning rate policy. The 1cycle policy anneals the learning
    rate from an initial learning rate to some maximum learning rate and then
    from that maximum learning rate to some minimum learning rate much lower
    than the initial learning rate.
    This policy was initially described in the paper `Super-Convergence:
    Very Fast Training of Neural Networks Using Large Learning Rates`_.

    .. note:
        This implementation is a refinement of :class:`torch.optim.OneCycleLR` that
        splits the learning rate decay into two pieces, one for returning to
        initial learning rate and another for decaying below the initial learning
        rate.


    The 1cycle learning rate policy changes the learning rate after every batch.
    `step` should be called after a batch has been used for training.

    This scheduler is not chainable.

    Note also that the total number of steps in the cycle can be determined in one
    of two ways (listed in order of precedence):

    #. A value for total_steps is explicitly provided.
    #. A number of epochs (epochs) and a number of steps per epoch
       (steps_per_epoch) are provided.
       In this case, the number of total steps is inferred by
       total_steps = epochs * steps_per_epoch

    You must either provide a value for total_steps or provide a value for both
    epochs and steps_per_epoch.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        max_lr (float or list): Upper learning rate boundaries in the cycle
            for each parameter group.
        total_steps (int): The total number of steps in the cycle. Note that
            if a value is not provided here, then it must be inferred by providing
            a value for epochs and steps_per_epoch.
            Default: None
        epochs (int): The number of epochs to train for. This is used along
            with steps_per_epoch in order to infer the total number of steps in the cycle
            if a value for total_steps is not provided.
            Default: None
        steps_per_epoch (int): The number of steps per epoch to train for. This is
            used along with epochs in order to infer the total number of steps in the
            cycle if a value for total_steps is not provided.
            Default: None
        pct_warmup (float): The percentage of the cycle (in number of steps) spent
            increasing the learning rate.
            Default: 0.3
        pct_cooldown (float): The percentage of the cycle (in number of steps) spent
            decreasing the learning rate to the initial learning rate.
            Default: 0.3
        anneal_strategy (str): {'cos', 'linear'}
            Specifies the annealing strategy: "cos" for cosine annealing, "linear" for
            linear annealing.
            Default: 'cos'
        cycle_momentum (bool): If ``True``, momentum is cycled inversely
            to learning rate between 'base_momentum' and 'max_momentum'.
            Default: True
        base_momentum (float or list): Lower momentum boundaries in the cycle
            for each parameter group. Note that momentum is cycled inversely
            to learning rate; at the peak of a cycle, momentum is
            'base_momentum' and learning rate is 'max_lr'.
            Default: 0.85
        max_momentum (float or list): Upper momentum boundaries in the cycle
            for each parameter group. Functionally,
            it defines the cycle amplitude (max_momentum - base_momentum).
            Note that momentum is cycled inversely
            to learning rate; at the start of a cycle, momentum is 'max_momentum'
            and learning rate is 'base_lr'
            Default: 0.95
        div_factor (float): Determines the initial learning rate via
            initial_lr = max_lr/div_factor
            Default: 25
        final_div_factor (float): Determines the minimum learning rate via
            min_lr = initial_lr/final_div_factor
            Default: 1e4
        last_epoch (int): The index of the last batch. This parameter is used when
            resuming a training job. Since `step()` should be invoked after each
            batch instead of after each epoch, this number represents the total
            number of *batches* computed, not the total number of epochs computed.
            When last_epoch=-1, the schedule is started from the beginning.
            Default: -1
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

    .. _Super-Convergence\: Very Fast Training of Neural Networks Using Large Learning Rates:
        https://arxiv.org/abs/1708.07120
    """

    def __init__(
        self,
        optimizer: Optimizer,
        max_lr: float,
        total_steps: Optional[int] = None,
        epochs: Optional[int] = None,
        steps_per_epoch: Optional[int] = None,
        pct_warmup: float = 0.3,
        pct_cooldown: float = 0.3,
        anneal_strategy: str = "cos",
        cycle_momentum: bool = True,
        base_momentum: float = 0.85,
        max_momentum: float = 0.95,
        div_factor: float = 25.0,
        final_div_factor: float = 1e4,
        last_epoch: int = -1,
        verbose: bool = False,
    ):
        super().__init__(
            optimizer,
            max_lr,
            total_steps,
            epochs,
            steps_per_epoch,
            pct_warmup,
            anneal_strategy,
            cycle_momentum,
            base_momentum,
            max_momentum,
            div_factor,
            final_div_factor,
            last_epoch,
            verbose,
        )
        if pct_warmup + pct_cooldown > 1.0:
            raise ValueError("Expected `pct_warmup` + `pct_cooldown` <= 1.0, " f"found {pct_warmup + pct_cooldown}")
        self.step_size_down = float(pct_cooldown * self.total_steps) - 1
        self.step_size_final = float(self.total_steps - self.step_size_up - self.step_size_down) - 1

    def __repr__(self):
        s = f"{self.__class__.__name__}(steps_up={self.step_size_up}"
        s += f", steps_down={self.step_size_down}"
        s += f", steps_final={self.step_size_final}"
        s += ")"
        return s

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, " "please use `get_last_lr()`.", UserWarning
            )

        lrs = []
        step_num = self.last_epoch

        if step_num > self.total_steps:
            raise ValueError(
                "Tried to step {} times. The specified number of total steps is {}".format(
                    step_num + 1, self.total_steps
                )
            )

        for group in self.optimizer.param_groups:
            # increasing lr phase
            if step_num <= self.step_size_up:
                computed_lr = self.anneal_func(group["initial_lr"], group["max_lr"], step_num / self.step_size_up)
                if self.cycle_momentum:
                    computed_momentum = self.anneal_func(
                        group["max_momentum"], group["base_momentum"], step_num / self.step_size_up
                    )

            # decreasing lr phase
            elif step_num <= self.step_size_up + self.step_size_down:
                down_step_num = step_num - self.step_size_up
                computed_lr = self.anneal_func(
                    group["max_lr"], group["initial_lr"], down_step_num / self.step_size_down
                )
                if self.cycle_momentum:
                    computed_momentum = self.anneal_func(
                        group["base_momentum"], group["max_momentum"], down_step_num / self.step_size_down
                    )

            # final decreasing lr phase
            else:
                final_step_num = step_num - (self.step_size_up + self.step_size_down)
                computed_lr = self.anneal_func(
                    group["initial_lr"], group["min_lr"], final_step_num / self.step_size_final
                )
                if self.cycle_momentum:
                    computed_momentum = group["base_momentum"]

            lrs.append(computed_lr)
            if self.cycle_momentum:
                if self.use_beta1:
                    _, beta2 = group["betas"]
                    group["betas"] = (computed_momentum, beta2)
                else:
                    group["momentum"] = computed_momentum

        return lrs
