#!/usr/bin/env python
# -*- coding: utf-8 -*-

from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Dict, Iterator, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from .common import MLP, BatchNormMixin, DropPath, SequenceBatchNorm


def duplicate(layer: nn.TransformerEncoderLayer) -> nn.TransformerEncoderLayer:
    r"""Duplicates all layers in a transformer except for self attention and feedforward"""
    new_layer = deepcopy(layer)
    new_layer.self_attn = layer.self_attn
    new_layer.linear1 = layer.linear1
    new_layer.linear2 = layer.linear2
    return new_layer


def entropy(x: Tensor, eps: float = 1e-2, dim: int = -1) -> Tensor:
    p = x
    log_p = x.log().clamp_min(-1 / eps)

    C = x.shape[dim]
    assert C > 0

    with torch.no_grad():
        divisor = p.new_tensor(C).log_()

    return log_p.mul(p).sum(dim=dim).neg().div(divisor).clamp(min=0, max=1)


@dataclass
class PerceiverLayerConfig:
    latent_d: int
    input_d: int
    latent_l: Optional[int] = None
    input_ff: Optional[int] = None
    latent_ff: Optional[int] = None
    nhead_latent: int = 1
    nhead_input: int = 1
    dropout: float = 0.0
    attn_dropout: float = 0.0
    act: nn.Module = nn.Mish()
    use_batchnorm: bool = False
    drop_path_rate: float = 0.0
    num_transformers: int = 1
    num_transformer_blocks: int = 1
    track_entropy: bool = False
    track_weights: bool = False
    feedforward_inputs: bool = True
    se_ratio: Optional[int] = None

    def instantiate(self) -> "PerceiverLayer":
        return PerceiverLayer(
            self.latent_d,
            self.input_d,
            self.latent_l,
            self.input_ff,
            self.latent_ff,
            self.nhead_latent,
            self.nhead_input,
            self.dropout,
            self.attn_dropout,
            self.act,
            self.use_batchnorm,
            self.drop_path_rate,
            self.num_transformers,
            self.num_transformer_blocks,
            self.track_entropy,
            self.track_weights,
            self.feedforward_inputs,
            self.se_ratio,
        )

    def replace(self, **kwargs) -> "PerceiverLayerConfig":
        return replace(self, **kwargs)


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

        attn_dropout:
            Dropout applied to the cross-attention matrices

        act:
            Activation for all MLP / feedforward layers

        use_batchnorm:
            If ``True``, disable all dropout and replace :class:`nn.LayerNorm` with :class:`SequenceBatchNorm`

        drop_path_rate:
            Rate for drop path / stochastic depth

        num_transformers:
            Number of latent transformer repeats in per transformer block. Transformers repeats within a block share
            weighs (iterative attention), aside from normalization layers which are unique to each repeat.
            See `BAIR`_ for justification.

        num_transformer_blocks:
            Number of unique blocks of latent transformer repeats in the layer.
            This balances expensive input/latent cross-attends versus cheap latent/latent self-attends.

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
        attn_dropout: float = 0.0,
        act: nn.Module = nn.Mish(),
        use_batchnorm: bool = False,
        drop_path_rate: float = 0.0,
        num_transformers: int = 1,
        num_transformer_blocks: int = 1,
        track_entropy: bool = False,
        track_weights: bool = False,
        feedforward_inputs: bool = True,
        se_ratio: Optional[int] = None,
    ):
        super().__init__()
        input_ff = input_ff or input_d
        latent_ff = latent_ff or latent_d
        self.track_entropy = track_entropy
        self.track_weights = track_weights

        if latent_l is not None:
            self.latent = nn.Parameter(torch.empty(latent_l, latent_d))
            nn.init.normal_(self.latent, 0, 1)
        else:
            self.latent = None

        # latent -> input cross attention
        self.cross_attn1 = nn.MultiheadAttention(
            latent_d,
            nhead_latent,
            kdim=input_d,
            vdim=input_d,
            dropout=attn_dropout,
        )
        self.norm_ca1 = nn.LayerNorm(latent_d)

        # input -> latent cross attention
        self.cross_attn2 = nn.MultiheadAttention(
            input_d, nhead_input, kdim=latent_d, vdim=latent_d, dropout=attn_dropout
        )
        self.norm_ca2 = nn.LayerNorm(input_d)

        # latent transformer blocks
        self.latent_transformer = nn.ModuleList()
        for _ in range(num_transformer_blocks):
            latent_transformer = nn.TransformerEncoderLayer(latent_d, nhead_latent, latent_ff, dropout)
            latent_transformer.activation = deepcopy(act)
            for _ in range(num_transformers):
                self.latent_transformer.append(duplicate(latent_transformer))

        # input feedforward blocks
        self.feedforward_inputs = feedforward_inputs
        if feedforward_inputs:
            self.ff = MLP(input_d, input_ff, dropout=dropout, act=deepcopy(act), se_ratio=se_ratio)
            self.norm_ff = nn.LayerNorm(input_d)

        self.drop_path = DropPath(drop_path_rate)

        if use_batchnorm:
            self.use_batchnorm(self)

        self.latent_w = None
        self.input_w = None

    def forward(self, inputs: Tensor, latent: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        Li, N, Di = inputs.shape
        latent = self.get_latent(inputs, latent)
        Ll, N, Dl = latent.shape
        orig_inputs, orig_latent = inputs, latent

        # Get attention mask (defaults to None)
        attn_mask = self.get_mask(inputs, latent)
        assert attn_mask is None or attn_mask.shape == (N, Ll, Li)
        attn_mask_T = attn_mask.transpose(-1, -2) if attn_mask is not None else None
        assert attn_mask_T is None or attn_mask.shape == (N, Li, Ll)

        # Cross attention 1; input -> latent
        need_weights = self.track_weights or self.track_entropy
        latent_attn, self.latent_w = self.cross_attn1(
            latent, inputs, inputs, need_weights=need_weights, attn_mask=attn_mask
        )
        latent = self.norm_ca1(latent + latent_attn)

        # Update
        if self.feedforward_inputs:
            inputs = self.norm_ff(inputs + self.ff(inputs))
        for transformer in self.latent_transformer:
            new_latent = transformer(latent)
            latent = self.drop_path(new_latent, latent)

        # Cross attention 2; latent -> input
        if self.feedforward_inputs:
            input_attn, self.input_w = self.cross_attn2(
                inputs, latent, latent, need_weights=self.track_entropy, attn_mask=attn_mask_T
            )
            inputs = self.norm_ca2(inputs + input_attn)

        # stochastic depth
        if self.feedforward_inputs:
            inputs = self.drop_path(inputs, orig_inputs)

        assert latent is not None
        return inputs, latent

    def get_latent(self, inputs: Tensor, latent: Optional[Tensor]) -> Tensor:
        if latent is not None:
            assert latent.ndim == 3
            assert latent.shape[1] == inputs.shape[1], f"{input.shape} vs {latent.shape}"
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

    @property
    def latent_entropy(self) -> Optional[Tensor]:
        r"""Get entropy of latent cross attention weights from last forward pass"""
        if self.latent_w is None:
            return None
        latent_w = self.latent_w
        N, Ll, Li = latent_w.shape
        return entropy(latent_w, dim=-1)

    @property
    def input_entropy(self) -> Optional[Tensor]:
        r"""Get entropy of input cross attention weights from last forward pass"""
        if self.input_w is None:
            return None
        input_w = self.input_w
        N, Li, Ll = input_w.shape
        return entropy(input_w, dim=-1)

    @staticmethod
    def perceiver_layers(model: nn.Module, prefix: str = "") -> Dict[str, "PerceiverLayer"]:
        r"""Gets all PerceiverLayers within ``model``.

        Args:
            model:
                The model to find PerceiverLayers in

            prefix:
                String prefix prepended to each dictionary key in the result
        """
        result: Dict[str, PerceiverLayer] = {}
        if isinstance(model, PerceiverLayer):
            result[prefix] = model
        for name, module in model.named_children():
            p = f"{prefix}.{name}" if prefix else name
            result.update(PerceiverLayer.perceiver_layers(module, prefix=p))
        return result

    @staticmethod
    def entropy_dict(model: nn.Module, reduce: bool = False) -> Dict[str, Tensor]:
        r"""Computes cross attention entropy for PerceiverLayers within ``model``.

        Args:
            model:
                The model to compute PerceiverLayer entropy for

            reduce:
                If ``True``, reduce the batch dimension and return a scalar

        Returns:
            Dictionary of layer name, entropy
        """
        result: Dict[str, Tensor] = {}
        for name, layer in PerceiverLayer.perceiver_layers(model).items():
            latent_entropy = layer.latent_entropy
            input_entropy = layer.input_entropy
            if latent_entropy is not None:
                result[f"{name}.latent"] = latent_entropy.mean(dim=-1) if not reduce else latent_entropy.mean()
            if input_entropy is not None:
                result[f"{name}.input"] = input_entropy.mean(dim=-1) if not reduce else input_entropy.mean()
        return result

    @staticmethod
    def regularizer(model: nn.Module, ord: Union[int, float] = 2, min: float = 0, max: float = 1) -> Tensor:
        r"""Computes a cross-attention entropy based regularizer for PerceiverLayers within ``model``.

        Args:
            model:
                The model to compute entropy regularizer for

            ord:
                Passed to :func:`torch.linalg.vector_norm` when reducing across multiple layers

            min:
                Minimum clamp value for entropy

            max:
                Maximum clamp value for entropy

        Returns:
            Scalar entropy regularizer
        """
        layer_entropy = PerceiverLayer.entropy_dict(model, reduce=True)
        num_layers = len(layer_entropy)
        if not num_layers:
            raise RuntimeError(f"No layers had Gini calculation enabled")
        ent = torch.stack([v for v in layer_entropy.values()], dim=0)
        ent = ent.clamp(min=min, max=max)
        ent = torch.linalg.vector_norm(ent, dim=0, ord=ord) / num_layers
        return ent.mean()

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

    @property
    def latent_layers(self) -> Iterator[nn.Module]:
        yield self.latent_transformer

    @property
    def norm_layers(self) -> Iterator[nn.Module]:
        for layer in self.children():
            if isinstance(layer, (nn.LayerNorm, SequenceBatchNorm)):
                yield layer

    @staticmethod
    def compute_kl_loss(latent: Tensor) -> Tensor:
        var, mean = torch.var_mean(latent, dim=(0, -1))
        kl = 0.5 * (mean.pow(2) + var - torch.log(var) - 1)
        return kl.mean()
