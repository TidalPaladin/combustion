#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
from typing import List, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim.optimizer import Optimizer

from typing import Optional, Tuple, Iterable, Dict, Any



import math
import torch
from torch.optim.optimizer import Optimizer
from copy import deepcopy
import numpy as np


class AdaHessian(Optimizer):
    """Implements Adahessian algorithm.
    It has been proposed in `ADAHESSIAN: An Adaptive Second Order Optimizer for Machine Learning`.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 0.15)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-4)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        hessian_power (float, optional): Hessian power (default: 1)
        spatial_average_block_size (int, optional): Spatial average block size for 1d tensors (default: (-1, -1, -1, -1) ). 
        Here for now, we only write down the tensor size from 1D to 4D. For higher dimension tensors (e.g., 5D), you can incorporate 
        the code by yourself. 
            -1 for 1D: no spatial average 
            -1 for 2D: use the entire row as the spatial average
            -1 for 3D (we assume it is a 1D Conv, you can customize it): use the channel (last dimension) of 1D Conv as spatial average
            -1 for 4D (we assume it is a 2D Conv, you can customize it): use the channel (last two dimension) of 2D Conv as spatial average
    """

    def __init__(self, params, lr=0.15, betas=(0.9, 0.999), eps=1e-4,
                 weight_decay=0, hessian_power=1, spatial_average_block_size=(-1, -1, -1, -1), single_gpu=True):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 0: {}".format(
                    betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 1: {}".format(
                    betas[1]))
        if not 0.0 <= hessian_power <= 1.0:
            raise ValueError("Invalid Hessian power value: {}".format(hessian_power))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, hessian_power=hessian_power)
        self.single_gpu = single_gpu 
        super(AdaHessian, self).__init__(params, defaults)

        self.spatial_average_block_size = spatial_average_block_size

    def get_params(self):
        """
        Gets all parameters in all param_groups with gradients
        """
        return (p for group in self.param_groups for p in group['params'] if p.requires_grad)

    def zero_hessian(self):
        """
        Zeros out the accumalated hessian traces.
        """

        for p in self.get_params():
            if not isinstance(p.hess, float): #and self.state[p]["hessian step"] % self.update_each == 0:
                p.hess.zero_()

    @torch.no_grad()
    def get_trace(self, params, grads):
        """
        compute the Hessian vector product with a random vector v, at the current gradient point,
        i.e., compute the gradient of <gradsH,v>.
        :param gradsH: a list of torch variables
        :return: a list of torch tensors
        """

        # Check backward was called with create_graph set to True
        for i, grad in enumerate(grads):
            if grad.grad_fn is None:
                raise RuntimeError('Gradient tensor {:} does not have grad_fn. When calling\n'.format(i) +
                           '\t\t\t  loss.backward(), make sure the option create_graph is\n' +
                           '\t\t\t  set to True.')

        v = [ 2 * torch.randint_like(p, high=2) - 1 for p in params]

        # this is for distributed setting with single node and multi-gpus, 
        # for multi nodes setting, we have not support it yet.
        if not self.single_gpu:
            for v1 in v:
                dist.all_reduce(v1)
        if not self.single_gpu:
            for v_i in v:
                v_i[v_i < 0.] = -1.
                v_i[v_i >= 0.] = 1.


        hvs = torch.autograd.grad(
            grads,
            params,
            grad_outputs=v,
            only_inputs=True,
            retain_graph=False)

        bs_1D, bs_2D, bs_3D, bs_4D = self.spatial_average_block_size

        hutchinson_trace = []
        for hv in hvs:
            param_size = hv.size()

            hv_abs = hv.abs()

            if len(param_size) <= 1:  
                # For 1D tensor, e.g.,, bias, BatchNorm, LayerNorm etc.
                # Usually, you do not need to set spatial aveging for it, i.e., Hessian diagonal block size is 1 here.
                
                if bs_1D == -1:
                    hutchinson_trace.append(hv_abs)
                else:
                    tmp_output1 = hv_abs.view(-1, bs_1D) # faltten to the N times bs_1D
                    tmp_output2 = torch.mean(tmp_output1, dim=[1])
                    tmp_output3 = tmp_output2.repeat_interleave(bs_1D).view(param_size)
                    hutchinson_trace.append(tmp_output3)

            elif len(param_size) == 2: 
                # For 2D tensor, e.g., the matrix in the fully-connected layer.
                # This is a normal case for MLP, Transformer models. 
                # Usually, a spatial averaging needs to be used here to get the best result.

                if bs_2D == -1:
                    hutchinson_trace.append( torch.mean(hv_abs, dim=[1], keepdim=True) )
                else:
                    tmp_output1 = hv_abs.view(-1, bs_2D) # faltten to the N times bs_2D
                    tmp_output2 = torch.mean(tmp_output1, dim=[1])
                    tmp_output3 = tmp_output2.repeat_interleave(bs_2D).view(param_size)
                    hutchinson_trace.append(tmp_output3)

            elif len(param_size) == 3:
                # For 3D tensor, e.g., the 1D Conv layer.
                # This layer is usually used for Char-LM.

                if bs_3D == -1:
                    hutchinson_trace.append( torch.mean(hv_abs, dim=[2], keepdim=True) )
                else:
                    tmp_output1 = hv_abs.view(-1, bs_3D) # faltten to the N times bs_3D
                    tmp_output2 = torch.mean(tmp_output1, dim=[1])
                    tmp_output3 = tmp_output2.repeat_interleave(bs_3D).view(param_size)
                    hutchinson_trace.append(tmp_output3)


            elif len(param_size) == 4:  
                # For 4D tensor, e.g, the 2D Conv layer
                # This layer is usually used for CV tasks.

                if bs_4D == -1:
                    hutchinson_trace.append( torch.mean(hv_abs, dim=[2, 3], keepdim=True) )
                else:
                    tmp_output1 = hv_abs.view(-1, bs_4D) # faltten to the N times bs_4D
                    tmp_output2 = torch.mean(tmp_output1, dim=[1])
                    tmp_output3 = tmp_output2.repeat_interleave(bs_4D).view(param_size)
                    hutchinson_trace.append(tmp_output3)

            else:
                raise RuntimeError(f'You need to write your customized function to support this shape: {param_size}')

        # this is for distributed setting with single node and multi-gpus, 
        # for multi nodes setting, we have not support it yet.
        if not self.single_gpu:
            for output1 in hutchinson_trace:
                dist.all_reduce(output1 / torch.cuda.device_count())

        return hutchinson_trace

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            gradsH: The gradient used to compute Hessian vector product.
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = torch.enable_grad()(closure)()

        params = []
        groups = []
        grads = []

        if not any(
            p.grad is not None and p.grad.grad_fn is not None
            for group in self.param_groups
            for p in group["params"]
        ):
            msg = (
                'Found no gradients that also have grad_fn. When '
                'calling loss.backward(), make sure the option '
                'create_graph is set to True.'
            )
            raise RuntimeError(msg)

        # Flatten groups into lists, so that
        #  hut_traces can be called with lists of parameters
        #  and grads
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad and p.grad is not None:
                    params.append(p)
                    groups.append(group)
                    grads.append(p.grad)

        # get the Hessian diagonal
        hut_traces = self.get_trace(params, grads)
        assert len(params) == len(groups) == len(grads) == len()

        #for group in self.param_groups:
        #    for i, p in enumerate(group['params']):
        #        if p.grad is None:
        #            continue

        for i, (p, group, grad, hut_trace) in enumerate(zip(
            params, groups, grads, hut_traces
        )):

                grad = deepcopy(grad.data)
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of Hessian diagonal square values
                    state['exp_hessian_diag_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_hessian_diag_sq = state['exp_avg'], state['exp_hessian_diag_sq']

                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_hessian_diag_sq.mul_(beta2).addcmul_(
                    1 - beta2, hut_trace, hut_trace)

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # make the square root, and the Hessian power
                k = group['hessian_power']
                denom = (
                    (exp_hessian_diag_sq.sqrt() ** k) /
                    math.sqrt(bias_correction2) ** k).add_(
                    group['eps'])

                # make update
                p.data = p.data - \
                    group['lr'] * (exp_avg / bias_correction1 / denom + group['weight_decay'] * p.data)

        return loss


class AdaHessian2(AdaHessian):

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            gradsH: The gradient used to compute Hessian vector product.
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = torch.enable_grad()(closure)()

        params = []
        groups = []
        grads = []

        if not any(
            p.grad is not None and p.grad.grad_fn is not None
            for group in self.param_groups
            for p in group["params"]
        ):
            msg = (
                'Found no gradients that also have grad_fn. When '
                'calling loss.backward(), make sure the option '
                'create_graph is set to True.'
            )
            raise RuntimeError(msg)

        # Flatten groups into lists, so that
        #  hut_traces can be called with lists of parameters
        #  and grads
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad and p.grad is not None:
                    params.append(p)
                    groups.append(group)
                    grads.append(p.grad)

        # get the Hessian diagonal
        hut_traces = self.get_trace(params, grads)

        #for group in self.param_groups:
        #    for i, p in enumerate(group['params']):
        #        if p.grad is None:
        #            continue

        for i, (p, group, grad, hut_trace) in enumerate(zip(
            params, groups, grads, hut_traces
        )):

                grad = deepcopy(grad.data)
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['mem'] = torch.ones_like(p.data)
                    state['avg_g'] = torch.zeros_like(p.data)
                    state['avg_g2'] = torch.zeros_like(p.data)
                    state['avg_h'] = torch.zeros_like(p.data)

                avg_g, avg_g2, avg_h, mem = (state[x] for x in ("avg_g", "avg_g2", "avg_h", "mem"))
                beta1, beta2 = group['betas']
                state['step'] += 1

                t = 1 / mem
                avg_g = (1 - t) * avg_g + t * grad
                avg_g2 = (1 - t) * avg_g2 + t * grad**2
                avg_h = (1 - t) * avg_h + t * abs(hut_trace)
                lrate = avg_g**2 / (avg_g2 + avg_h)
                lrate = lrate.clamp_max(1)
                mem = (1 - grad**2/avg_g2) * mem + 1
                with torch.no_grad():
                    print(lrate.mean())

                state["avg_g"] = avg_g
                state["avg_g2"] = avg_g2
                state["avg_h"] = avg_h
                state["mem"] = mem

                # make update
                p.data = p.data - group['lr'] * (grad + group['weight_decay'] * p.data)

        return loss




class AdaHessianMixin:
    "Wrap `opt` in a AdaHessian optimizer"

    param_groups: Iterable[Dict[str, Any]]
    block_length: int = 32
    update_each: int = 1
    n_samples: int = 1

    def get_params(self, with_grads: bool = False) -> Tuple[nn.Parameter]:
        """
        Gets all parameters in all param_groups with gradients
        """
        return tuple(p for group in self.param_groups for p in group['params'] if p.requires_grad) # type: ignore


    def zero_hessian(self):
        """
        Zeros out the accumalated hessian traces.
        """
        for p in self.get_params():
            if not isinstance(p.hess, float) and self.state[p]["hessian step"] % self.update_each == 0:
                p.hess.zero_()


    @torch.no_grad()
    def set_hessian(self, params: Optional[List[Tensor]]=None, grads: Optional[List[Tensor]]=None):
        """
        Computes the Hutchinson approximation of the hessian trace and accumulates it for each trainable parameter.
        """
        if params is None:
            params = []
            for p in filter(lambda p: p.grad is not None, self.get_params()):
                if self.state[p]["hessian step"] % self.update_each == 0:  # compute the trace only each `update_each` step
                    params.append(p)
                self.state[p]["hessian step"] += 1
                params.append(p)

        assert params is not None
        grads = grads or [p.grad for p in params]
        assert grads is not None
        assert len(grads) == len(params)

        if len(params) == 0:
            import pdb; pdb.set_trace()
            return

        if self.generator.device != params[0].device:  # hackish way of casting the generator to the right device
            self.generator = torch.Generator(params[0].device).manual_seed(2147483647)

        #p, g = [], []
        #for param, grad in zip(params, grads):
        #    if grad is not None:
        #        p.append(param)
        #        g.append(grad)
        #params = p
        #grads = g

        for i in range(self.n_samples):
            zs = [torch.randint(0, 2, p.size(), generator=self.generator, device=p.device) * 2.0 - 1.0 for p in params]  # Rademacher distribution {-1.0, 1.0}
            h_zs = torch.autograd.grad(grads, params, grad_outputs=zs, only_inputs=True, retain_graph=False)
            for h_z, z, p in zip(h_zs, zs, params):
                p.hess += h_z * z / self.n_samples  # approximate the expected values of z*(H@z)
                if p.hess.isnan().any():
                    import pdb; pdb.set_trace()



class AdaHessian(Optimizer, AdaHessianMixin):
    """Implements Adahessian algorithm.
    It has been proposed in `ADAHESSIAN: An Adaptive Second Order Optimizer for Machine Learning`.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 0.15)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-4)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        hessian_power (float, optional): Hessian power (default: 1)
        spatial_average_block_size (int, optional): Spatial average block size for 1d tensors (default: (-1, -1, -1, -1) ). 
        Here for now, we only write down the tensor size from 1D to 4D. For higher dimension tensors (e.g., 5D), you can incorporate 
        the code by yourself. 
            -1 for 1D: no spatial average 
            -1 for 2D: use the entire row as the spatial average
            -1 for 3D (we assume it is a 1D Conv, you can customize it): use the channel (last dimension) of 1D Conv as spatial average
            -1 for 4D (we assume it is a 2D Conv, you can customize it): use the channel (last two dimension) of 2D Conv as spatial average
    """

    def __init__(self, params, lr=0.15, betas=(0.9, 0.999), eps=1e-4,
                 weight_decay=0, hessian_power=1, spatial_average_block_size=(-1, -1, -1, -1), single_gpu=True):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 0: {}".format(
                    betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 1: {}".format(
                    betas[1]))
        if not 0.0 <= hessian_power <= 1.0:
            raise ValueError("Invalid Hessian power value: {}".format(hessian_power))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, hessian_power=hessian_power)
        self.single_gpu = single_gpu 
        super(AdaHessian, self).__init__(params, defaults)

        # use a separate generator that deterministically generates the same `z`s across all GPUs in case of distributed training
        self.generator = torch.Generator().manual_seed(2147483647)

        self.spatial_average_block_size = spatial_average_block_size
        self.block_length = 32
        self.average_conv_kernel = False

        for p in self.get_params():
            p.hess = 0.0
            self.state[p]["hessian step"] = 0

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.
        Arguments:
            closure (callable, optional) -- a closure that reevaluates the model and returns the loss (default: None)
        """

        loss = None
        if closure is not None:
            loss = torch.enable_grad()(closure)()

        self.zero_hessian()
        self.set_hessian()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None or p.hess is None:
                    continue

                if self.average_conv_kernel and p.dim() == 4:
                    p.hess = torch.abs(p.hess).mean(dim=[2, 3], keepdim=True).expand_as(p.hess).clone()

                # Perform correct stepweight decay as in AdamW
                p.mul_(1 - group['lr'] * group['weight_decay'])

                state = self.state[p]

                # State initialization
                if len(state) == 1:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)  # Exponential moving average of gradient values
                    state['exp_hessian_diag_sq'] = torch.zeros_like(p.data)  # Exponential moving average of Hessian diagonal square values

                exp_avg, exp_hessian_diag_sq = state['exp_avg'], state['exp_hessian_diag_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(p.grad, alpha=1 - beta1)
                exp_hessian_diag_sq.mul_(beta2).addcmul_(p.hess, p.hess, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                bias_correction1 = 1 
                #bias_correction2 = 1

                k = group['hessian_power']
                denom = (exp_hessian_diag_sq / bias_correction2).pow_(k / 2).add_(0.1)

                # make update
                step_size = group['lr'] / bias_correction1
                p.addcdiv_(exp_avg, denom, value=-step_size)


        return loss


        t = 1/mem
        avg_g = (1 - t) * avg_g + t * grad
        avg_g2 = (1 - t) * avg_g2 + t * grad**2
        avg_h = (1 - t) * avg_h + t * abs(h)
        lrate = avg_g**2 / (avg_g2 + avg_h)
        mem = (1 - g**2/avg_g2) * mem + 1
        param = param - lrate * grad


class AdaHessian(Optimizer, AdaHessianMixin):
    """Implements Adahessian algorithm.
    It has been proposed in `ADAHESSIAN: An Adaptive Second Order Optimizer for Machine Learning`.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 0.15)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-4)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        hessian_power (float, optional): Hessian power (default: 1)
        spatial_average_block_size (int, optional): Spatial average block size for 1d tensors (default: (-1, -1, -1, -1) ). 
        Here for now, we only write down the tensor size from 1D to 4D. For higher dimension tensors (e.g., 5D), you can incorporate 
        the code by yourself. 
            -1 for 1D: no spatial average 
            -1 for 2D: use the entire row as the spatial average
            -1 for 3D (we assume it is a 1D Conv, you can customize it): use the channel (last dimension) of 1D Conv as spatial average
            -1 for 4D (we assume it is a 2D Conv, you can customize it): use the channel (last two dimension) of 2D Conv as spatial average
    """

    def __init__(self, params, lr=0.15, betas=(0.9, 0.999, 0.99), eps=1e-4,
                 weight_decay=0, hessian_power=1, spatial_average_block_size=(-1, -1, -1, -1), single_gpu=True):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 0: {}".format(
                    betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 1: {}".format(
                    betas[1]))
        if not 0.0 <= hessian_power <= 1.0:
            raise ValueError("Invalid Hessian power value: {}".format(hessian_power))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, hessian_power=hessian_power)
        self.single_gpu = single_gpu 
        super(AdaHessian, self).__init__(params, defaults)

        # use a separate generator that deterministically generates the same `z`s across all GPUs in case of distributed training
        self.generator = torch.Generator().manual_seed(2147483647)

        self.spatial_average_block_size = spatial_average_block_size
        self.block_length = 32
        self.average_conv_kernel = False

        for p in self.get_params():
            p.hess = 0.0
            self.state[p]["hessian step"] = 0

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.
        Arguments:
            closure (callable, optional) -- a closure that reevaluates the model and returns the loss (default: None)
        """

        loss = None
        if closure is not None:
            loss = torch.enable_grad()(closure)()

        self.zero_hessian()
        self.set_hessian()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None or p.hess is None:
                    continue

                if self.average_conv_kernel and p.dim() == 4:
                    p.hess = torch.abs(p.hess).mean(dim=[2, 3], keepdim=True).expand_as(p.hess).clone()

                # Perform correct stepweight decay as in AdamW
                p.mul_(1 - group['lr'] * group['weight_decay'])

                state = self.state[p]

                # State initialization
                if len(state) == 1:
                    state['step'] = 0
                    state['avg_g'] = torch.zeros_like(p.data)  # Exponential moving average of gradient values
                    state['avg_g2'] = torch.zeros_like(p.data)  # Exponential moving average of Hessian diagonal square values
                    state['avg_h'] = torch.zeros_like(p.data)  # Exponential moving average of Hessian diagonal square values

                avg_g, avg_g2, avg_h, mem = (state[x] for x in ("avg_g", "avg_g2", "avg_h", "mem"))

                beta1, beta2, beta3 = group['betas']
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                bias_correction3 = 1 - beta3 ** state['step']

                avg_g = (1 - beta1) * avg_g + beta1 * p.grad
                avg_g2 = (1 - beta2) * avg_g2 + beta2 * p.grad**2
                avg_h = (1 - beta3) * avg_h + beta3 * abs(p.hess)

                #avg_g = avg_g / bias_correction1
                #avg_g2 = avg_g2 / bias_correction2
                #avg_h = avg_h / bias_correction3

                if mem.isnan().any():
                    import pdb; pdb.set_trace()
                if avg_g.isnan().any():
                    import pdb; pdb.set_trace()

                assert not avg_g.isnan().any()
                assert not avg_g2.isnan().any()
                assert not avg_h.isnan().any()

                state["avg_g"] = avg_g
                state["avg_g2"] = avg_g2
                state["avg_h"] = avg_h
                state["mem"] = mem


                lrate = avg_g**2 / (avg_g2 + avg_h + 1e-4)
                lrate.clamp_max_(1.0)
                p = p - lrate * p.grad

        return loss
