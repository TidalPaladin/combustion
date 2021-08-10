#!/usr/bin/env python
# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from abc import ABC, abstractmethod
from enum import IntEnum, Enum
from dataclasses import dataclass, field

from torch import Tensor
from typing import Any, Callable, Optional, Tuple, List, Type
from math import sqrt
from functools import partial
from .fnet import FNet
from copy import deepcopy
from .common import MLP, SqueezeExcite, SequenceBatchNorm, SequenceInstanceNorm, BatchNormMixin, DropPath


def duplicate(layer: nn.TransformerEncoderLayer) -> nn.TransformerEncoderLayer:
    r"""Duplicates all layers in a transformer except for self attention and feedforward"""
    new_layer = deepcopy(layer)
    new_layer.self_attn = layer.self_attn
    new_layer.linear1 = layer.linear1
    new_layer.linear2 = layer.linear2
    return new_layer


class ConstantLatent(nn.Module, BatchNormMixin):

    def __init__(self, latent_d: int, input_d, latent_l: int, dim_ff: Optional[int] = None, act: nn.Module = nn.ReLU(), use_batchnorm: bool = False, dropout: float = 0.1):
        super().__init__()
        self.latent = nn.Parameter(torch.empty(latent_l, latent_d))
        nn.init.xavier_uniform_(self.latent)

        # NOTE: layer has unstable gradients and trains poorly at higher LR when batch normalized
        #if use_batchnorm:
        #    self.use_batchnorm(self)

    def extra_repr(self) -> str:
        s = f"latent_d={self.latent_d}, input_d={self.input_d}, latent_l={self.latent_l}"
        return s

    def forward(self, inputs: Tensor) -> Tensor:
        N = inputs.shape[1]
        L, D = self.latent.shape
        return self.latent.view(L, 1, D).expand(-1, N, -1).contiguous()


@dataclass
class PerceiverBlockConfig:
    latent_d: int
    input_d: int
    dim_ff: Optional[int] = None
    nhead_latent: int = 1
    nhead_input: int = 1
    dropout: float = 0.0
    dual_latent: bool = False
    act: nn.Module = nn.ReLU()
    init_ff: Optional[int] = None
    use_batchnorm: bool = False
    drop_path_rate: float = 0.1
    num_transformers: int = 1
    repetitions: int = 1
    mixer: nn.Module = nn.Identity()
    favor: bool = False
    performer: bool = False

    def instantiate(self) -> nn.Module: 
        if self.dual_latent:
            return PerceiverDualLatent.from_config(self)
        else:
            return PerceiverLayer.from_config(self)

    def initializer(self, latent_l: int, input_d: Optional[int] = None) -> nn.Module: 
        input_d = input_d or self.input_d
        return InitialLatent(self.latent_d, input_d, latent_l, self.init_ff, self.act, self.use_batchnorm)


class PerceiverLayer(nn.Module, BatchNormMixin):

    def __init__(
        self, 
        latent_d: int, 
        input_d: int, 
        latent_l: Optional[int] = None,
        input_ff: Optional[int] = None, 
        latent_ff: Optional[int] = None, 
        nhead_latent: int = 1, 
        nhead_input: int = 1, 
        dropout: float = 0.0,
        act: nn.Module = nn.Mish(),
        use_batchnorm: bool = False,
        drop_path_rate: float = 0.1,
        num_transformers: int = 1,
        track_entropy: bool = False,
        track_weights: bool = False,
    ):
        super().__init__()
        input_ff = input_ff or input_d
        latent_ff = latent_ff or latent_d
        self.track_entropy = track_entropy
        self.track_weights = track_weights

        if latent_l is not None:
            self.latent = nn.Parameter(torch.empty(latent_l, 1, latent_d))
            nn.init.normal_(self.latent, mean=0, std=1)
        else:
            self.latent = None

        self.cross_attn1 = nn.MultiheadAttention(latent_d, nhead_latent, kdim=input_d, vdim=input_d,)
        self.norm_ca1 = nn.LayerNorm(latent_d)

        self.cross_attn2 = nn.MultiheadAttention(input_d, nhead_input, kdim=latent_d, vdim=latent_d)
        self.norm_ca2 = nn.LayerNorm(input_d)

        # latent transformer blocks
        latent_transformer = nn.TransformerEncoderLayer(latent_d, nhead_latent, latent_ff, dropout)
        latent_transformer.activation = deepcopy(act)
        self.latent_transformer = nn.Sequential(*[duplicate(latent_transformer)]*num_transformers)

        # input feedforward blocks
        self.ff1 = MLP(input_d, input_ff, dropout=dropout, act=deepcopy(act))
        self.ff2 =  MLP(input_d, input_ff, dropout=dropout, act=deepcopy(act))
        self.norm_ff1 = nn.LayerNorm(input_d)
        self.norm_ff2 = nn.LayerNorm(input_d)

        self.drop_path = DropPath(drop_path_rate)

        if use_batchnorm:
            self.use_batchnorm(self)

    def forward(self, inputs: Tensor, latent: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        L_i, N, D_i = inputs.shape
        orig_inputs = inputs
        latent = latent if latent is not None else self.latent
        L_l, _, D_l = latent.shape
        if latent.shape[1] != N:
            latent = latent.expand(L_l, N, D_l)
        orig_latent = latent

        if latent is None:
            if self.latent is None:
                raise RuntimeError(f"Must pass a latent when self.latent is None")
            latent = self.latent
        assert latent is not None

        # Cross attention 1; input -> latent
        need_weights = self.track_weights or self.track_entropy
        latent_attn, latent_w = self.cross_attn1(latent, inputs, inputs, need_weights=need_weights)
        latent = self.norm_ca1(latent + latent_attn)

        # Update 
        inputs = self.norm_ff1(inputs + self.ff1(inputs))
        latent = self.latent_transformer(latent)

        # Cross attention 2; latent -> input
        input_attn, input_w = self.cross_attn2(inputs, latent, latent, need_weights=self.track_entropy)
        inputs = self.norm_ca2(inputs + input_attn)

        # move forward
        inputs = self.norm_ff2(inputs + self.ff2(inputs))

        # stochastic depth
        sd_mask = self.drop_path.get_mask(inputs)
        inputs = orig_inputs * (~sd_mask) + inputs * sd_mask
        latent = orig_latent * (~sd_mask) + latent * sd_mask

        assert latent is not None
        return inputs, latent

    @classmethod
    def from_config(cls, conf: PerceiverBlockConfig) -> "PerceiverLayer": 
        return cls(
            conf.latent_d, 
            conf.input_d, 
            conf.dim_ff, 
            conf.nhead_latent, 
            conf.nhead_input, 
            conf.dropout, 
            conf.act, 
            conf.use_batchnorm, 
            conf.drop_path_rate,
            conf.num_transformers,
            conf.repetitions,
            conf.mixer,
            conf.favor,
            conf.performer,
        )



class PerceiverFPN(nn.Module):

    def __init__(self, configs: List[PerceiverBlockConfig], eps: float = 1e-4):
        super().__init__()
        self.perceivers = nn.ModuleList([x.instantiate() for x in configs])
        num_joins = len(configs)
        self.weights = nn.Parameter(torch.ones(num_joins, 2))
        self.eps = eps

    def forward(self, tensors: List[Tensor]) -> List[Tensor]:
        assert len(tensors) == len(self.perceivers) + 1

        all_inputs: List[Optional[Tensor]] = [None for x in tensors]
        all_latents: List[Optional[Tensor]] = [None for x in tensors]
        for i, perceiver in enumerate(self.perceivers):
            inputs = tensors[i]
            latent = tensors[i+1]

            if i > 0 and i < len(self.perceivers) - 1:
                weight0, weight1 = self.weights[i - 1]
            else:
                weight0, weight1 = 1, 1

            latent, inputs = perceiver(latent, inputs)
            all_inputs[i] = inputs
            all_latents[i+1] = latent

        out_tensors: List[Tensor] = []
        for i, (inputs, latent) in enumerate(zip(all_inputs, all_latents)):
            if inputs is None and latent is not None:
                out_tensors.append(latent)
            elif latent is None and inputs is not None:
                out_tensors.append(inputs)
            else:
                assert i > 0 
                assert i < len(self.weights) + 1, str(i)
                w1, w2 = self.weights[i-1].relu()
                w_input = w1 / (w1 + w2)
                w_latent = w2 / (w1 + w2)
                out = w_input * inputs + w_latent * latent
                out_tensors.append(out)

        assert len(out_tensors) == len(tensors)
        return out_tensors
