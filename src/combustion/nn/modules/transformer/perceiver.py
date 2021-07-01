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
from .fnet import FourierDownsample
from copy import deepcopy

class SequenceBatchNorm(nn.BatchNorm1d):

    def forward(self, x: Tensor) -> Tensor:
        x = x.movedim(0, -1).contiguous()
        x = super().forward(x)
        x = x.movedim(-1, 0).contiguous()
        return x


class BatchNormMixin:

    @staticmethod
    def use_batchnorm(module: nn.Module, **kwargs):
        for name, layer in module.named_children():
            if hasattr(layer, "dropout") and isinstance(layer.dropout, float):
                layer.dropout = 0
            if isinstance(layer, nn.LayerNorm):
                d = layer.normalized_shape[0]
                new_layer = SequenceBatchNorm(d, **kwargs)
                #new_layer = nn.LazyBatchNorm1d(**kwargs)
                setattr(module, name, new_layer)
            elif isinstance(layer, nn.Dropout):
                new_layer = nn.Identity()
                setattr(module, name, new_layer)
            else:
                BatchNormMixin.use_batchnorm(layer)


class DropPath(nn.Module):
    def __init__(self, ratio: float):
        super().__init__()
        self.ratio = 1.0 - abs(float(ratio))
        assert self.ratio >= 0 and self.ratio < 1.0

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or not self.ratio:
            return x

        L, N, D = x.shape
        with torch.no_grad():
            mask = self.ratio + torch.rand(N).type_as(x).floor_()
            mask = mask.view(1, N, 1)

        assert mask.ndim == x.ndim
        assert mask.shape[1] == N
        return x / self.ratio * mask


class MLP(nn.Module):
    def __init__(self, d: int, d_hidden: int, dropout: float = 0, act: nn.Module = nn.ReLU()):
        super().__init__()
        self.l1 = nn.Linear(d, d_hidden)
        self.d1 = nn.Dropout(dropout)
        self.l2 = nn.Linear(d_hidden, d)
        self.d2 = nn.Dropout(dropout)
        self.act = act

    def forward(self, x: Tensor) -> Tensor:
        x = self.l1(x)
        x = self.act(x)
        x = self.d1(x)
        x = self.l2(x)
        x = self.act(x)
        x = self.d2(x)
        return x


class SqueezeExcite(nn.Module):

    def __init__(self, d_in, d_squeeze, act: nn.Module = nn.ReLU()):
        super().__init__()
        self.se = nn.Sequential(
            nn.Linear(d_in, d_squeeze),
            act,
            nn.Linear(d_squeeze, d_in),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        weights = self.se(x.mean(dim=0, keepdim=True))
        #x = x + x * weights
        return x * weights


class InitialLatent(nn.Module, BatchNormMixin):

    def __init__(self, latent_d: int, input_d, latent_l: int, dim_ff: Optional[int] = None, act: nn.Module = nn.ReLU(), use_batchnorm: bool = False, dropout: float = 0.1):
        super().__init__()
        self.dim_ff = dim_ff or input_d
        self.input_d = input_d
        self.latent_d = latent_d
        self.latent_l = latent_l

        self.linear1 = nn.Sequential(
            nn.Linear(input_d, self.dim_ff),
            deepcopy(act),
        )
        self.linear2 = nn.Linear(self.dim_ff, latent_d*latent_l)
        self.final_act = nn.Sequential(
            nn.Dropout(dropout),
            deepcopy(act),
            nn.LayerNorm(latent_d),
        )

        # NOTE: layer has unstable gradients and trains poorly at higher LR when batch normalized
        #if use_batchnorm:
        #    self.use_batchnorm(self)

    def extra_repr(self) -> str:
        s = f"latent_d={self.latent_d}, input_d={self.input_d}, latent_l={self.latent_l}"
        return s

    def forward(self, inputs: Tensor) -> Tensor:
        L, N, D = inputs.shape
        D_l, L_l = self.latent_d, self.latent_l
        x = inputs.mean(dim=0)
        x = self.linear1(x)
        x = self.linear2(x)
        x = x.view(N, L_l, D_l).movedim(0, 1).contiguous()
        x = self.final_act(x)
        return x

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
        dim_ff: Optional[int] = None, 
        nhead_latent: int = 1, 
        nhead_input: int = 1, 
        dropout: float = 0.0,
        act: nn.Module = nn.ReLU(),
        use_batchnorm: bool = False,
        drop_path_rate: float = 0.1
    ):
        super().__init__()
        dim_ff = dim_ff or input_d
        self.cross_attn1 = nn.MultiheadAttention(latent_d, nhead_latent, kdim=input_d, vdim=input_d)
        self.cross_attn2 = nn.MultiheadAttention(input_d, nhead_input, kdim=latent_d, vdim=latent_d)

        self.norm_ca1 = nn.LayerNorm(latent_d)
        self.norm_ca2 = nn.LayerNorm(input_d)

        self.latent_transformer = nn.TransformerEncoderLayer(latent_d, nhead_latent, dim_ff, dropout)
        self.latent_transformer.activation = deepcopy(act)

        self.ff1 = nn.Sequential(
            MLP(input_d, input_d, dropout, act=deepcopy(act)),
            SqueezeExcite(input_d, input_d // 4, act=deepcopy(act))
        )
        self.norm_ff1 = nn.LayerNorm(input_d)


        self.ff2 = nn.Sequential(
            MLP(input_d, input_d, act=deepcopy(act)),
            SqueezeExcite(input_d, input_d // 4, act=deepcopy(act))
        )
        self.norm_ff2 = nn.LayerNorm(input_d)

        self.latent_norm = nn.LayerNorm(latent_d)
        #self.latent_ff = MLP(latent_d, latent_d, act=deepcopy(act))
        self.drop_path = DropPath(drop_path_rate)

        if use_batchnorm:
            self.use_batchnorm(self)

    def forward(self, latent: Tensor, inputs: Tensor) -> Tuple[Tensor, Tensor]:
        L_l, N, D_l = latent.shape
        L_i, N, D_i = inputs.shape
        orig_latent, orig_inputs = latent, inputs

        # move forward
        inputs = self.norm_ff1(inputs + self.ff1(inputs))
        latent = self.latent_transformer(latent)

        # cross attention 1
        latent_attn, _ = self.cross_attn1(latent, inputs, inputs, need_weights=False)
        latent = self.norm_ca1(latent + latent_attn)

        # move forward
        inputs = self.norm_ff2(inputs + self.ff2(inputs))
        #latent = self.latent_norm(latent + self.latent_ff(latent))

        # cross attention 2
        input_attn, _ = self.cross_attn2(inputs, latent, latent, need_weights=False)
        inputs = self.norm_ca2(inputs + input_attn)

        # stochastic depth
        inputs = orig_inputs + self.drop_path(inputs)
        latent = orig_latent + self.drop_path(latent)

        return latent, inputs

    @classmethod
    def from_config(cls, conf: PerceiverBlockConfig) -> "PerceiverLayer": 
        return cls(conf.latent_d, conf.input_d, conf.dim_ff, conf.nhead_latent, conf.nhead_input, conf.dropout, conf.act, conf.use_batchnorm, conf.drop_path_rate)


class PerceiverDualLatent(nn.Module, BatchNormMixin):

    def __init__(
        self, 
        d1: int, 
        d2: int, 
        nhead1: int = 1, 
        nhead2: int = 1, 
        dropout: float = 0.0,
        act: nn.Module = nn.ReLU(),
        use_batchnorm: bool = False,
        drop_path_rate: float = 0.1
    ):
        super().__init__()
        self.cross_attn1 = nn.MultiheadAttention(d1, nhead1, kdim=d2, vdim=d2)
        self.cross_attn2 = nn.MultiheadAttention(d2, nhead2, kdim=d1, vdim=d1)

        self.norm_ca1 = nn.LayerNorm(d1)
        self.norm_ca2 = nn.LayerNorm(d2)

        self.t1 = nn.TransformerEncoderLayer(d1, nhead1, d1, dropout)
        self.t1.activation = deepcopy(act)
        self.t2 = nn.TransformerEncoderLayer(d2, nhead2, d2, dropout)
        self.t2.activation = deepcopy(act)

        self.drop_path = DropPath(drop_path_rate)

        if use_batchnorm:
            self.use_batchnorm(self)


    def forward(self, latent: Tensor, inputs: Tensor) -> Tuple[Tensor, Tensor]:
        orig_latent, orig_inputs = latent, inputs

        latent = self.t1(latent)
        inputs = self.t2(inputs)

        # cross attention 1
        latent_attn, _ = self.cross_attn1(latent, inputs, inputs, need_weights=False)
        latent = self.norm_ca1(latent + latent_attn)

        # cross attention 2
        inputs_attn, _ = self.cross_attn2(inputs, latent, latent, need_weights=False)
        inputs = self.norm_ca2(inputs + inputs_attn)

        # stochastic depth
        inputs = orig_inputs + self.drop_path(inputs)
        latent = orig_latent + self.drop_path(latent)

        return latent, inputs

    @classmethod
    def from_config(cls, conf: PerceiverBlockConfig) -> "PerceiverDualLatent": 
        return cls(conf.latent_d, conf.input_d, conf.nhead_latent, conf.nhead_input, conf.dropout, conf.act, conf.use_batchnorm, conf.drop_path_rate)



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




class PerceiverBlock(nn.Module):

    def __init__(
        self,
        conf: PerceiverBlockConfig,
        latent_init: InitialLatent,
        repeats: int = 1
    ):
        super().__init__()
        self.input_d = conf.input_d
        self.latent_d = conf.latent_d

        self.initializer = latent_init
        self.layers = nn.ModuleList()
        for _ in range(repeats):
            layer = conf.instantiate()
            self.layers.append(layer)

        if not conf.dual_latent:
            self.pos_emb_linears = nn.ModuleList()
            for _ in range(repeats + 1):
                linear = nn.Sequential(
                    nn.Linear(conf.input_d, conf.input_d),
                    deepcopy(conf.act)
                )
                self.pos_emb_linears.append(linear)
        else:
            self.pos_emb_linears = None

    def forward(self, inputs: Tensor, pos_emb: Optional[Tensor] = None, init_inputs: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        if pos_emb is not None:
            inputs = inputs + self.pos_emb_linears[0](pos_emb)

        init_inputs = init_inputs if init_inputs is not None else inputs
        latent = self.initializer(init_inputs)
        features = inputs

        for i, block in enumerate(self.layers):
            latent, features = block(latent, features)
            if pos_emb is not None:
                linear = self.pos_emb_linears[i+1]
                features = features + linear(pos_emb)
                
        return latent, features


@dataclass
class PerceiverConfig:
    levels: List[PerceiverBlockConfig] = field(default_factory=lambda: [
        PerceiverBlockConfig(64, 32, nhead_latent=2, nhead_input=2, act=nn.SiLU()),
        PerceiverBlockConfig(128, 64, nhead_latent=2, nhead_input=2, dual_latent=True, act=nn.SiLU(), init_ff=32),
        PerceiverBlockConfig(256, 128, nhead_latent=4, nhead_input=4, dual_latent=True, act=nn.SiLU(), init_ff=32)
    ])
    latent_l: List[int] = field(default_factory=lambda: [32, 16, 8])
    repeats: List[int] = field(default_factory=lambda : [3, 2, 2])
    fpn_repeats: int = 0
    init_from_input: bool = False
    use_batchnorm: bool = False


class Perceiver(nn.Module):

    def __init__(self, levels: List[PerceiverBlockConfig], latent_l: List[int], repeats: List[int], fpn_repeats: int = 0, init_from_input: bool = False, use_batchnorm: bool = False):
        super().__init__()
        self.init_from_input = init_from_input
        if use_batchnorm:
            for level in levels:
                level.use_batchnorm = True

        self.blocks = nn.ModuleList()
        for level, ll, repeat in zip(levels, latent_l, repeats):
            initializer_d = levels[0].input_d if init_from_input else None
            initializer = level.initializer(ll, initializer_d)
            block = PerceiverBlock(level, initializer, repeat)
            self.blocks.append(block)

        fpn = nn.ModuleList()
        for _ in range(fpn_repeats):
            block = PerceiverFPN(levels)
            fpn.append(block)
        if fpn_repeats:
            self.fpn = fpn
        else:
            self.fpn = None

    def forward(self, inputs: Tensor, pos_emb: Optional[Tensor] = None) -> List[Tensor]:
        result: List[Tensor] = []
        orig_input = inputs
        for i, block in enumerate(self.blocks):
            emb = pos_emb if i == 0 else None
            initializer_input = orig_input if self.init_from_input else None
            latent, inputs = block(inputs, emb, initializer_input)
            result.append(inputs)
            inputs = latent
        result.append(latent)

        if self.fpn is not None:
            for block in self.fpn:
                result = block(result)

        return result

    @classmethod
    def from_config(cls, conf: PerceiverConfig) -> "Perceiver":
        return cls(conf.levels, conf.latent_l, conf.repeats, conf.fpn_repeats, conf.init_from_input, conf.use_batchnorm)

    @property
    def num_features(self) -> List[int]:
        input_d = self.blocks[0].input_d
        return [input_d] + [x.latent_d for x in self.blocks]
