#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch import Tensor
from typing import Any, Callable, Optional, Tuple, List
from math import sqrt
from functools import partial

KernelFunction = Callable[[Tensor, Tensor], Tensor]

@torch.no_grad()
def make_orthogonal(mat: Tensor, uniform_q = False) -> Tensor:
    r"""Forces a matrix of random features to be orthogonal via Gram-Schmidt renormalization 
    (implemented via QR decomposition). This process will maintain unbiasedness when ``mat``
    contains random features sampled from an isotropic distribution.
    """
    R, C = mat.shape[-2:]
    if R != C:
        raise ValueError(f"Expected rows == cols, found {R}, {C}")

    # QR decomposition gives orthogonal matrix
    q, r = torch.linalg.qr(mat, mode="reduced")

    # proposed by @Parskatt
    # to make sure Q is uniform https://arxiv.org/pdf/math-ph/0609050.pdf
    if uniform_q:
        d = torch.diagonal(r, 0, -2, -1).unsqueeze_(-1)
        q.mul_(d.sign_())
    return q

@torch.no_grad()
def gaussian_orthogonal_random_matrix(
    rows: int, 
    cols: int, 
    uniform_q = False, 
    scaling: int = 1,
    **kwargs
) -> Tensor:
    r"""Creates a matrix of Gaussian orthogonal random features.

    The matrix is created by:
        1. Sampling values from a standard normal distribution
        2. Forcing these features to be orthogonal via QR decomposition 
           (Gram-Schmidt renormalization)
        3. Applying scaling

    Args:
        rows:
            Number of rows in the created matrix
            
        cols:
            Number of columns in the created matrix

        uniform_q:
            If ``True``, make sure Q (from QR decomposition) is uniform

        scaling:
            TODO

    Keyword Args:
        Passed to tensor constructors (e.g. ``device`` or ``dtype``)

    Shape:
        Output - :math:`(R, C)` where :math:`R, C` are ``rows``, ``cols``.
    """
    blocks = rows // cols + (1 if rows % cols else 0)
    unstructured_block = torch.randn((blocks, cols, cols), **kwargs)

    m = unstructured_block.view(-1, cols)[:rows].clone()
    m = m.norm(dim=-1)

    # make_orthogonal uses QR decomposition to return orthogonal columns.
    # apply transpose to create orthogonal rows
    q = (
        make_orthogonal(unstructured_block, uniform_q)
        .transpose(-2, -1)
        .view(-1, cols)
    )

    q = q[:rows]
    assert tuple(q.shape) == (rows, cols)

    # TODO this should be an enum or something
    if scaling == 0:
        #multiplier = torch.randn((rows, cols), **kwargs).norm(dim = -1)
        multiplier = m
    elif scaling == 1:
        multiplier = q.new_full((rows,), fill_value=sqrt(float(cols)))
    else:
        raise ValueError(f'Invalid scaling {scaling}')

    q = multiplier.diag() @ q
    return q


def generalized_kernel_features(
    data: Tensor,
    kernel_func: KernelFunction,
    projection: Optional[Tensor] = None,
    normalize: bool = True,
    kernel_epsilon: float = 1e-3
):
    r"""Constructs kernel features for fast generalized attention.

    Shapes:
        * ``data`` - :math:`(L, N, E)`
        * ``projection`` - :math:`(E, R)`
        Output - :math:`(L, N, R)` if ``projection`` is not ``None``, otherwise :math:`(L, N, E)`
    """

    L, N, E = data.shape

    normalizer = (E ** -0.25) if normalize else 1
    data = data * normalizer

    if projection is None:
        return kernel_func(data, data) + kernel_epsilon
    
    R = projection.shape[-1]
    projection = projection.view(1, E, R).expand(N, -1, -1)
    assert projection.shape == (N, E, R)
    assert data.shape == (L, N, E)

    proj_data = torch.einsum("lne,ner->lnr", data, projection)
    assert proj_data.shape == (L, N, R)

    out = kernel_func(data, proj_data) + kernel_epsilon
    return out

def _softmax_kernel(data: Tensor, proj_data: Tensor, is_query: bool, normalizer: float, ratio: float) -> Tensor:
    # h(x) = exp(-||x||**2 / 2)
    # NOTE: x will have been subject to 1 / E ** 4 normalizer, so undo that
    h = (
        data
        .pow(2)
        .sum(dim=-1, keepdim=True)
        .div(2)
        .mul(normalizer ** 2)
    )

    if is_query:
        diff = proj_data.amax(dim=-1, keepdim=True)
    else:
        diff = proj_data.amax(dim=(0, -1), keepdim=True)

    return ratio * torch.exp(proj_data - h - diff)

def softmax_kernel(data, projection, is_query, normalize=True, kernel_eps=1e-6):
    r"""Implements an approximation of the softmax (SM) kernel using positve random features.

    The softmax kernel can be approximated as

    .. math:
        \phi(x) = \frac{h(x)}{\sqrt{m}}

    """
    L, N, E = data.shape
    _, R = projection.shape

    normalizer = (E ** -0.25) if normalize else 1. # root(d) normalization
    #ratio = (R ** -0.5) # 1 / root(m) normalization from FAVOR
    ratio = (R ** -0.5) # 1 / root(m) normalization from FAVOR

    #projection = projection.view(1, E, R).expand(N, -1, -1)
    #assert projection.shape == (N, E, R)
    #assert data.shape == (L, N, E)

    kernel_fn = partial(
        _softmax_kernel, 
        normalizer=normalizer,
        ratio=ratio,
        is_query=is_query, 
    )

    return generalized_kernel_features(
        data,
        kernel_fn,
        projection,
        normalize,
        kernel_epsilon = kernel_eps * ratio
    )

# non-causal linear attention
def linear_attention(
    q: Tensor, 
    k: Tensor, 
    v: Tensor, 
    fast: bool = True, 
    return_weights: bool = False,
    dropout: float = 0.0,
    stabilizer: float = 1e-4
) -> Tuple[Tensor, Optional[Tensor]]:
    L, N, E = v.shape
    v = v.permute(1, 0, 2)
    kT = k.permute(1, 2, 0)
    q = q.permute(1, 0, 2)
    _, R, _ = kT.shape

    assert tuple(v.shape) == (N, L, E)
    assert tuple(q.shape) == (N, L, R)
    assert tuple(kT.shape) == (N, R, L)
    weight: Optional[Tensor] = None

    # NOTE: no root d normalization here, its baked into the kernel
    if fast:
        # D_inv = diag(Q'(K_t' @ 1_L))
        # here we express kT @ 1_L as kT.sum(dim=-1))
        # also, diag(a_i, ...).inverse() == diag(1 / a_i, ...)
        D_inv = q.bmm(kT.sum(dim=-1, keepdim=True)).view(N, L, 1)
        D_inv = D_inv + 2 * stabilizer * (D_inv.abs() <= stabilizer)
        D_inv = D_inv.reciprocal()

        context = kT @ v
        out = D_inv * (q @ context)

        # normally we wouldn't compute this since it isn't needed to compute performer attn
        if return_weights:
            weight = D_inv * (q @ kT)

    # Fallback to normal attn (for debugging)
    else:
        q = q / math.sqrt(E)
        weight = (q @ kT).softmax(dim=-1)
        out = weight @ v

    out = out.permute(1, 0, 2)
    assert out.shape == (L, N, E)
    assert weight is None or tuple(weight.shape) == (N, L, L)
    return out, weight

def _kernel_fn(_: Tensor, proj: Tensor) -> Tensor:
    return F.relu(proj)

class FAVOR(nn.MultiheadAttention):
    r"""Implements Fast Attention Via positive Orthogonal Random features (FAVOR+).
    FAVOR+ models kernelizable attention mechanisms (such as softmax attention) with
    provable accuracy at a linear (as opposed to quadratic) complexity.
    """
    projection_matrix: Optional[Tensor]
    
    def __init__(
        self, 
        embed_dim: int, 
        proj_dim: int, 
        num_heads: int, 
        dropout: float = 0., 
        bias: bool = True, 
        add_zero_attn: bool = False,
        uniform_q: bool = True,
        scaling: int = 0,
        fast: bool = True,
        generalized: bool = False,
        kernel_fn: KernelFunction = _kernel_fn,
        stabilizer: float = 1e-6
    ) -> None:
        super(FAVOR, self).__init__(
            embed_dim, 
            num_heads, 
            dropout, 
            bias, 
            False, 
            add_zero_attn,
        )
        self.proj_dim = proj_dim
        self.uniform_q = uniform_q
        self.scaling = scaling
        self.generalized = generalized
        self.kernel_fn = kernel_fn
        self.fast = fast
        self.stabilizer = stabilizer

        if self.fast:
            self.proj_head_dim = proj_dim // num_heads 
            assert self.proj_head_dim * num_heads == self.proj_dim, "proj_dim must be divisible by num_heads"
        else:
            self.proj_head_dim = self.head_dim

        projection_matrix = self.create_projection()
        self.register_buffer('projection_matrix', projection_matrix)

    def create_projection(self, **kwargs) -> Tensor:
        r"""Creates a projection matrix using positive Gaussian orthogonal random features."""
        mat = gaussian_orthogonal_random_matrix(
            self.embed_dim, 
            self.proj_dim, 
            self.uniform_q, 
            self.scaling, 
            **kwargs
        )
        return mat

    @torch.no_grad()
    def redraw_projection_matrix(self, **kwargs) -> None:
        r"""Redraws the projection matrix and places it into the buffer"""
        projections = self.create_projection(**kwargs)
        self.projection_matrix.copy_(projections)

    def create_softmax_kernel(self, data: Tensor, is_query: bool, **kwargs) -> Tensor:
        kernel = softmax_kernel(
            data, 
            self.projection_matrix, 
            is_query, 
            kernel_eps=self.stabilizer,
            **kwargs
        )
        return kernel

    def create_general_kernel(self, data: Tensor, **kwargs) -> Tensor:
        kernel = generalized_kernel_features(
            data, 
            self.kernel_fn,
            self.projection_matrix,  
            kernel_epsilon=self.stabilizer,
            **kwargs
        )
        return kernel

    def forward(
        self, 
        query: Tensor, 
        key: Tensor, 
        value: Tensor, 
        need_weights: bool = False,
        **kwargs
    ) -> Tuple[Tensor, Optional[Tensor]]:
        # apply initial mapping
        L, N, E = query.shape
        E_mh = self.head_dim
        needs_favor = L >= self.head_dim

        Q, K, V = self._in_projection_packed(query, key, value, self.in_proj_weight, self.in_proj_bias)

        # get kernel function mapping Q -> Q', ...
        if not self.fast or not needs_favor:
            Q = Q
            K = K
        elif self.generalized:
            Q = self.create_general_kernel(Q)
            K = self.create_general_kernel(K)
        else:
            Q = self.create_softmax_kernel(Q, is_query=True)
            K = self.create_softmax_kernel(K, is_query=False)

        # multi-head split
        L, N, R = query.shape
        Q = Q.reshape(L, -1, self.proj_head_dim)
        K = K.reshape(L, -1, self.proj_head_dim)
        V = V.reshape(L, -1, self.head_dim)

        out, weight = linear_attention(Q, K, V, self.fast, need_weights, stabilizer=self.stabilizer)
        out = out.reshape(L, N, -1)
        if weight is not None:
            with torch.no_grad():
                weight = weight.reshape(N, self.num_heads, L, L).sum(dim=1).div_(self.num_heads)

        out = self.out_proj(out)
        out = F.dropout(out, self.dropout, training=self.training)
        return out, weight

    @staticmethod
    def _in_projection_packed(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        w: Tensor,
        b: Optional[Tensor] = None,
    ) -> List[Tensor]:
        r"""Taken from F.multi_head_attention_forward"""
        E = q.size(-1)
        if k is v:
            if q is k:
                # self-attention
                return F.linear(q, w, b).chunk(3, dim=-1) # type: ignore
            else:
                # encoder-decoder attention
                w_q, w_kv = w.split([E, E * 2])
                if b is None:
                    b_q = b_kv = None
                else:
                    b_q, b_kv = b.split([E, E * 2])
                return (F.linear(q, w_q, b_q),) + F.linear(k, w_kv, b_kv).chunk(2, dim=-1) # type: ignore
        else:
            w_q, w_k, w_v = w.chunk(3)
            if b is None:
                b_q = b_k = b_v = None
            else:
                b_q, b_k, b_v = b.chunk(3)
            return F.linear(q, w_q, b_q), F.linear(k, w_k, b_k), F.linear(v, w_v, b_v) # type: ignore

                
class PerformerLayer(nn.TransformerEncoderLayer):
    r"""Implements the Performer layer as described in PAPER.
    The Performer uses Fast Attention Via positive Orthogonal Random features (FAVOR+) to
    reduce attention to a linear complexity operation. FAVOR is provided via :class:`FAVOR`.
    """
    redraw_step: Tensor
    
    def __init__(
        self, 
        d_model: int, 
        nhead: int, 
        favor_features: int, 
        dim_feedforward: int = 2048, 
        dropout: float = 0.1, 
        activation: str = "relu",
        feature_redraw_interval: int = 1000,
        auto_redraw_projections: bool = True,
        fast: bool = True,
        stabilizer: float = 1e-6
     ):
        super(PerformerLayer, self).__init__(
            d_model, 
            nhead, 
            dim_feedforward, 
            dropout, 
            activation
        )
        self.self_attn = FAVOR(d_model, favor_features, nhead, fast=fast, stabilizer=stabilizer)
        self.feature_redraw_interval = int(feature_redraw_interval)
        self.auto_redraw_projections = bool(auto_redraw_projections)
        self.activation = F.silu

        self.register_buffer("redraw_step", torch.tensor(0))
        if fast:
            self.register_forward_pre_hook(PerformerLayer._projection_redraw_hook)

    @staticmethod
    def _projection_redraw_hook(module: "PerformerLayer", *args, **kwargs) -> None:
        if not module.training:
            return 

        module.redraw_step.add_(1)
        needs_redraw = (
            module.auto_redraw_projections
            and module.redraw_step >=  module.feature_redraw_interval
        )
        if needs_redraw:
            module.self_attn.redraw_projection_matrix()
            module.redraw_step.fill_(0)
