#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from abc import ABC, abstractmethod
from enum import IntEnum, Enum

from torch import Tensor
from typing import Any, Callable, Optional, Tuple, List, Type
from math import sqrt
from functools import partial


class OrthoScaling(Enum):
    NORM = 0
    CONSTANT = 1
    NONE = 2


@torch.no_grad()
def make_orthogonal(mat: Tensor, uniform_q = True) -> Tensor:
    r"""Forces a matrix of random features to be orthogonal via Gram-Schmidt renormalization 
    (QR decomposition). This process will maintain unbiasedness when ``mat``
    contains random features sampled from an isotropic distribution.

    Args:
        mat:
            Input matrix to make orthogonal via QR decomposition

        uniform_q:
            If ``True``, ensure that orthogonal matrix :math:`Q` is unqiue.
            
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
    uniform_q = True, 
    scaling: OrthoScaling = OrthoScaling.NORM,
    **kwargs
) -> Tensor:
    r"""Creates a matrix of Gaussian orthogonal random features. The returned
    matrix will be (mostly) orthogonal over its rows.

    The matrix is created by:
        1. Sampling values from a standard normal distribution
        2. Forcing these features to be orthogonal via QR decomposition 
           (Gram-Schmidt renormalization)
        3. (Possibly) forcing the orthogonal matrix to be unique
        4. Applying scaling

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
    D, R = rows, cols
    assert R <= D, f"ORF requires R <= D, found {R}, {D}"
    blocks = rows // cols + (1 if rows % cols else 0)
    unstructured_block = torch.randn((blocks, cols, cols), **kwargs)

    # make_orthogonal uses QR decomposition to return orthogonal columns.
    # apply transpose to create orthogonal rows
    q = (
        make_orthogonal(unstructured_block, uniform_q)
        .contiguous()
        .view(-1, cols)
        .transpose(-2, -1)
    )
    q = q[..., :rows]
    assert tuple(q.shape) == (R, D)

    if scaling == OrthoScaling.NORM:
        #multiplier = unstructured_block.view(-1, cols)[:rows].norm(dim=-1)
        multiplier = unstructured_block.view(-1, cols)[:rows].norm(dim=-1).view(1, -1)
    elif scaling == OrthoScaling.CONSTANT:
        multiplier = D ** 0.5
    elif scaling == OrthoScaling.NONE:
        return q
    q.mul_(multiplier)

    return q

class KernelORF(nn.Module, ABC):
    r"""Base class for modules that approximate kernels using ORF

    Args:
        d:
            Dimensionality of the input

        num_features:
            Number of orthogonal random features to use in the approximation

        kernel_eps:
            Stabilizer added to the kernel

        normalize:
            If true, normalize input data using :math:`\sqrt{d}` normalization.

        uniform_q:
            If true, ensure ORF matrix is unique 

        scaling:
            Scaling applied to ORF

        feature_redraw_interval:
            Interval at which to redraw a new ORF matrix. If ``None``, do not automatically
            redraw the ORF matrix.

    Shape:
        * ``data`` - :math:`(L, N, E)`
        * Output - :math:`(L, N, F)` where :math:`F` is kernel-dependent
    """
    projection_matrix: Tensor
    redraw_step: Tensor

    def __init__(
        self, 
        d: int, 
        num_features: int, 
        kernel_eps: float = 1e-6, 
        normalize: bool = True,
        uniform_q: bool = True,
        scaling: OrthoScaling = OrthoScaling.CONSTANT,
        feature_redraw_interval: Optional[int] = 1000,
    ):
        super().__init__()
        self.kernel_eps = kernel_eps
        self.normalize = normalize
        self.d = d
        self.num_features = num_features
        self.uniform_q = uniform_q
        self.scaling = scaling

        projection_matrix = self.create_projection()
        self.register_buffer('projection_matrix', projection_matrix)

        self.feature_redraw_interval = int(feature_redraw_interval or 0)

        self.register_buffer("redraw_step", torch.tensor(0))
        self.register_forward_pre_hook(self.__class__._projection_redraw_hook)

    def extra_repr(self) -> str:
        s = ", ".join(
            f"{k}={getattr(self, k)}" for k in ("d", "num_features", "normalize", "uniform_q", "scaling")
        )
        if self.kernel_eps:
            s += f", {self.kernel_eps}"
        if not self.auto_redraw_features:
            s += f", auto_redraw_features=False"
        return s

    def forward(self, data: Tensor, **kwargs) -> Tensor:
        L, N, E = data.shape
        projection = self.projection_matrix

        normalizer = (E ** -0.25) if self.normalize else 1
        data = data * normalizer

        R = projection.shape[-2]
        projection = projection.view(1, *projection.shape).expand(N, -1, -1)
        data = data.permute(1, -1, 0)
        assert projection.shape == (N, R, E)
        assert data.shape == (N, E, L)

        proj_data = projection.bmm(data)
        assert proj_data.shape == (N, R, L)

        out = self.kernel_function(data, proj_data, **kwargs)
        out = out.permute(2, 0, 1)
        assert tuple(out.shape[:2]) == (L, N)
        return out

    @abstractmethod
    def kernel_function(self, data: Tensor, projection: Tensor, **kwargs) -> Tensor:
        r"""Defines the function :math:`\phi` for approximating a given kernel.

        Args:
            data:
                Input data, with :math:`\sqrt{d}` normalization applied if requested

            projection:
                Projected data, i.e. ``data @ self.projection_matrix``

        Returns:
            Output of :math:`\phi{\bf{x}}`
        """
        ...

    @torch.no_grad()
    def create_projection(self, **kwargs) -> Tensor:
        r"""Creates a projection matrix using positive Gaussian orthogonal random features."""
        mat = gaussian_orthogonal_random_matrix(
            self.d, 
            self.num_features, 
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

    @staticmethod
    def _projection_redraw_hook(module: "KernelORF", *args, **kwargs) -> None:
        if not module.training or not module.feature_redraw_interval:
            return 

        module.redraw_step.add_(1)
        if module.redraw_step >= module.feature_redraw_interval:
            module.redraw_projection_matrix()
            module.redraw_step.fill_(0)

class SoftmaxORF(KernelORF):

    def kernel_function(self, data: Tensor, projection: Tensor, is_query: bool) -> Tensor:
        N, E, L = data.shape
        _, R, _ = projection.shape # (N, R, L)
        E_dim = -2
        L_dim = -1

        ratio = (R ** -0.5) # 1 / root(m) normalization from FAVOR

        # h(x) = exp(-||x||**2 / 2)
        # TODO: why is the normalizer ** 2 factor necessary?
        # it seems to reduce MSE / max error. Maybe related to ortho scaling
        # applied to input ORF?
        h = (
            data
            .pow(2)
            .sum(dim=E_dim, keepdim=True)
            .div(2)
        )

        # additional stabilization
        if is_query:
            diff = projection.amax(dim=E_dim, keepdim=True)
        else:
            diff = projection.amax(dim=(E_dim, L_dim), keepdim=True)

        result =  ratio * torch.exp(projection - h - diff)
        return result

class SoftmaxHyp(KernelORF):

    def kernel_function(self, data: Tensor, projection: Tensor, is_query: bool) -> Tensor:
        N, E, L = data.shape
        _, R, _ = projection.shape # (N, R, L)
        E_dim = -2
        L_dim = -1

        ratio = ((2*R) ** -0.5) # 1 / root(2m) normalization 

        # h(x) = exp(-||x||**2 / 2)
        h = (
            data
            .pow(2)
            .sum(dim=E_dim, keepdim=True)
            .div(2)
        )

        projection = torch.cat((projection, projection.neg()), dim=E_dim)

        if is_query:
            diff = projection.amax(dim=E_dim, keepdim=True)
        else:
            diff = projection.amax(dim=(E_dim, L_dim), keepdim=True)

        result = ratio * torch.exp(projection - h - diff)
        return result


# non-causal linear attention
def linear_attention(
    q: Tensor, 
    k: Tensor, 
    v: Tensor, 
    fast: bool = True, 
    return_weights: bool = False,
    dropout: float = 0.0,
    stabilizer: float = 1e-6
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
        D_inv = (
            q.bmm(kT.sum(dim=-1, keepdim=True))
            .view(N, L, 1)
            .clamp_min(stabilizer)
            .reciprocal()
        )

        context = kT @ v
        out = D_inv * (q @ context)

        # normally we wouldn't compute this since it isn't needed to compute performer attn
        if return_weights:
            weight = D_inv * (q @ kT)

    # Fallback to normal attn (for debugging)
    else:
        q = q / (E ** -0.5)
        weight = q.bmm(kT).softmax(dim=-1)
        out = weight.bmm(v)

    out = out.permute(1, 0, 2)
    assert out.shape == (L, N, E)
    assert weight is None or tuple(weight.shape) == (N, L, L)
    return out, weight


class FAVOR(nn.MultiheadAttention):
    r"""Implements Fast Attention Via positive Orthogonal Random features (FAVOR+).
    FAVOR+ models kernelizable attention mechanisms (such as softmax attention) with
    provable accuracy at a linear (as opposed to quadratic) complexity.

    .. note:
        The required number of random features ``proj_dim`` to maintain a certain error 
        level grows with embedding size :math:`d` as :math:`d\log d`.
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
        fast: bool = True,
        kernel_fn: Type[KernelORF] = SoftmaxHyp,
        stabilizer: float = 1e-6,
        **kwargs
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
        self.fast = fast
        self.stabilizer = stabilizer

        if self.fast:
            self.proj_head_dim = proj_dim // num_heads 
            assert self.proj_head_dim * num_heads == self.proj_dim, "proj_dim must be divisible by num_heads"
        else:
            self.proj_head_dim = self.head_dim

        self.kernel = kernel_fn(self.head_dim, self.proj_head_dim, **kwargs)

    def forward(
        self, 
        query: Tensor, 
        key: Tensor, 
        value: Tensor, 
        need_weights: bool = False,
        fast: Optional[bool] = None,
        **kwargs
    ) -> Tuple[Tensor, Optional[Tensor]]:
        fast = bool(fast) if fast is not None else self.fast

        # apply initial mapping
        L, N, E = query.shape
        E_mh = self.head_dim
        #needs_favor = L >= self.head_dim

        # NOTE: explicitly call MHSA forward to get exact results when not using FAVOR. 
        # MHSA seems to use a C++ implementation and will only agree with pure torch implementation
        # within some (pretty good) level of precision
        if not fast:
            return super().forward(query, key, value, need_weights=need_weights, **kwargs)

        Q, K, V = self._in_projection_packed(query, key, value, self.in_proj_weight, self.in_proj_bias)

        # multi-head split
        L, N, R = query.shape
        Q = Q.contiguous().view(L, N * self.num_heads, -1)
        K = K.contiguous().view(L, N * self.num_heads, -1)
        V = V.contiguous().view(L, N * self.num_heads, -1)

        # get kernel function mapping Q -> Q', ...
        Q = self.kernel(Q, is_query=True)
        K = self.kernel(K, is_query=False)

        # attention
        out, weight = linear_attention(Q, K, V, self.fast, need_weights, stabilizer=self.stabilizer)
        out = out.contiguous().view(L, N, -1)

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
        self.self_attn = FAVOR(
            d_model, 
            favor_features, 
            nhead, 
            fast=fast, 
            stabilizer=stabilizer, 
            feature_redraw_interval=feature_redraw_interval
        )
