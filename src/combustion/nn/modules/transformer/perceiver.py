#!/usr/bin/env python
# -*- coding: utf-8 -*-


from copy import deepcopy
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .common import MLP, BatchNormMixin, DropPath


def duplicate(layer: nn.TransformerEncoderLayer) -> nn.TransformerEncoderLayer:
    r"""Duplicates all layers in a transformer except for self attention and feedforward"""
    new_layer = deepcopy(layer)
    new_layer.self_attn = layer.self_attn
    new_layer.linear1 = layer.linear1
    new_layer.linear2 = layer.linear2
    return new_layer


class PerceiverLayer(nn.Module, BatchNormMixin):
    r"""Implements the Perceiver as described in `Perceiver`_ and `Perceiver IO`_. The Perceiver
    is a general perception architecture that uses cross-attention with a fixed size latent space
    to extract features. It aims to be general w.r.t. its inputs and outputs, performing well across
    domains like images, video, point clouds, and audio.

    .. image:: ./perceiver.png
        :width: 800px
        :align: center
        :height: 300px
        :alt: Diagram of a Perceiver layer.

    Args:
        latent_d:
            Channel size of the latent space

        input_d:
            Channel size of the input space

        latent_l:
            Number of latent vectors. If ``None``, :func:`forward` will expect a latent
            to be passed to it manually.

        input_ff:
            Size of the hidden layer for input feedforward layers. Defaults to ``input_d``

        latent_ff:
            Size of the hidden layer for latent feedforward layers. Defaults to ``latent_d``

        nhead_latent:
            Number of heads for the latent transformer / cross-attention

        nhead_input:
            Number of heads for the input cross attention

        dropout:
            Dropout rate for all MLP / feedforward layers

        act:
            Activation for all MLP / feedforward layers

        use_batchnorm:
            If ``True``, disable all dropout and replace :class:`nn.LayerNorm` with :class:`SequenceBatchNorm`

        drop_path_rate:
            Rate for drop path / stochastic depth

        num_transformers:
            Number of latent transformer repeats in the layer. This balances expensive input/latent cross-attends
            versus cheap latent/latent self-attends. Latent transformers share weights (iterative attention), aside
            from normalization layers which are unique to each repeat. See `BAIR`_ for justification.

    Returns:
        Tuple of ``(transformed_inputs, transformed_latent)``

    Shapes:
        * ``inputs`` - :math:`(L_i, N, D_i)`
        * ``latent`` - :math:`(L_l, N, D_l)`
        * Output - same as inputs

    .. _BAIR: https://bair.berkeley.edu/blog/2021/03/23/universal-computation/
    .. _Perceiver: https://arxiv.org/abs/2103.03206
    .. _Perceiver IO: https://arxiv.org/abs/2107.14795
    """

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
        feedforward_inputs: bool = True,
    ):
        super().__init__()
        input_ff = input_ff or input_d
        latent_ff = latent_ff or latent_d
        self.track_entropy = track_entropy
        self.track_weights = track_weights

        if latent_l is not None:
            self.latent = nn.Parameter(torch.empty(latent_l, latent_d))
            nn.init.kaiming_normal_(self.latent, mode="fan_out")
        else:
            self.latent = None

        self.cross_attn1 = nn.MultiheadAttention(
            latent_d,
            nhead_latent,
            kdim=input_d,
            vdim=input_d,
        )
        self.norm_ca1 = nn.LayerNorm(latent_d)

        self.cross_attn2 = nn.MultiheadAttention(input_d, nhead_input, kdim=latent_d, vdim=latent_d)
        self.norm_ca2 = nn.LayerNorm(input_d)

        # latent transformer blocks
        latent_transformer = nn.TransformerEncoderLayer(latent_d, nhead_latent, latent_ff, dropout)
        latent_transformer.activation = deepcopy(act)
        self.latent_transformer = nn.ModuleList([duplicate(latent_transformer) for _ in range(num_transformers)])

        # input feedforward blocks
        self.feedforward_inputs = feedforward_inputs
        if feedforward_inputs:
            self.ff1 = MLP(input_d, input_ff, dropout=dropout, act=deepcopy(act))
            self.ff2 = MLP(input_d, input_ff, dropout=dropout, act=deepcopy(act))
            self.norm_ff1 = nn.LayerNorm(input_d)
            self.norm_ff2 = nn.LayerNorm(input_d)

        self.drop_path = DropPath(drop_path_rate)

        if use_batchnorm:
            self.use_batchnorm(self)

    def forward(self, inputs: Tensor, latent: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        latent = self.get_latent(inputs, latent)
        orig_inputs, orig_latent = inputs, latent
        attn_mask = self.get_mask(inputs, latent)
        attn_mask_T = attn_mask.transpose(-1, -2) if attn_mask is not None else None

        # Cross attention 1; input -> latent
        need_weights = self.track_weights or self.track_entropy
        latent_attn, latent_w = self.cross_attn1(latent, inputs, inputs, need_weights=need_weights, attn_mask=attn_mask)
        latent = self.norm_ca1(latent + latent_attn)

        # Update
        if self.feedforward_inputs:
            inputs = self.norm_ff1(inputs + self.ff1(inputs))
        for transformer in self.latent_transformer:
            latent = transformer(latent)

        # Cross attention 2; latent -> input
        if self.feedforward_inputs:
            input_attn, input_w = self.cross_attn2(
                inputs, latent, latent, need_weights=self.track_entropy, attn_mask=attn_mask_T
            )
            inputs = self.norm_ca2(inputs + input_attn)

        # move forward
        if self.feedforward_inputs:
            inputs = self.norm_ff2(inputs + self.ff2(inputs))

        # stochastic depth
        sd_mask = self.drop_path.get_mask(inputs)
        if self.feedforward_inputs:
            inputs = orig_inputs * (~sd_mask) + inputs * sd_mask
        latent = orig_latent * (~sd_mask) + latent * sd_mask

        assert latent is not None
        return inputs, latent

    def get_latent(self, inputs: Tensor, latent: Optional[Tensor]) -> Tensor:
        if latent is not None:
            assert latent.ndim == 3
            assert latent.shape[1] == inputs.shape[1]
            return latent
        else:
            if self.latent is None:
                raise RuntimeError(f"Must pass a latent when self.latent is None")
            latent = self.latent
            _, N, _ = inputs.shape
            latent = self.latent
            Ll, Dl = latent.shape
            latent = latent.view(Ll, 1, Dl).expand(-1, N, -1)
            return latent

    def get_mask(self, inputs: Tensor, latent: Tensor) -> Optional[Tensor]:
        r"""Override this to use masked attention"""
        return None

    def duplicate(self) -> "PerceiverLayer":
        new = deepcopy(self)
        # don't duplicate these layers
        shared = ["cross_attn1", "cross_attn2", "latent_transformer", "ff1", "ff2", "latent"]

        for name in shared:
            layer = getattr(self, name, None)
            if layer is None:
                continue
            if name == "latent_transformer":
                new_layer = nn.ModuleList([duplicate(t) for t in layer])
            else:
                new_layer = layer
            setattr(new, name, new_layer)

        return new
