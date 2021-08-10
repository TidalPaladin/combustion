#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from abc import ABC, abstractmethod
from enum import IntEnum, Enum

from torch import Tensor
from typing import Any, Callable, Optional, Tuple, List, Type, Union, Dict 
from math import sqrt
from functools import partial
from .common import MLP, DropPath
from .point_transformer import KNNTail, KNNDownsample
from .experimental import Unattention, SoftmaxSE
from copy import deepcopy


class OrthoScaling(Enum):
    NORM = 0
    CONSTANT = 1
    NONE = 2


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
    mat = mat.transpose(-2, -1)

    # QR decomposition gives orthogonal matrix
    q, r = torch.linalg.qr(mat, mode="reduced")

    # proposed by @Parskatt
    # to make sure Q is uniform https://arxiv.org/pdf/math-ph/0609050.pdf
    if uniform_q:
        d = torch.diagonal(r, 0, -2, -1).unsqueeze(-1)
        q = q.mul(d.sign())
    return q.transpose(-2, -1)

@torch.no_grad()
def gaussian_orthogonal_random_matrix(
    size: int, 
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
    N = size
    unstructured_block = torch.randn((N, N), **kwargs)
    multiplier = torch.linalg.vector_norm(unstructured_block, dim=-1)

    # make_orthogonal uses QR decomposition to return orthogonal columns.
    # apply transpose to create orthogonal rows
    q = (
        make_orthogonal(unstructured_block, uniform_q)
        .contiguous()
    )
    return multiplier.diag().matmul(q)


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
        num_heads: int, 
        kernel_eps: float = 1e-6, 
        normalize: bool = True,
        uniform_q: bool = True,
        scaling: OrthoScaling = OrthoScaling.CONSTANT,
        feature_redraw_interval: Optional[int] = 1000,
        trainable: bool = True
    ):
        super().__init__()
        if trainable:
            feature_redraw_interval = None

        self.kernel_eps = kernel_eps
        self.normalize = normalize
        self.d = d
        self.num_heads = num_heads
        self.num_features = self.d // self.num_heads
        self.uniform_q = uniform_q
        self.scaling = scaling
        self.trainable = trainable
        self.feature_redraw_interval = int(feature_redraw_interval or 0)

        projection_matrix = self.create_projection()
        if trainable:
            self.projection_matrix = nn.Parameter(projection_matrix, requires_grad=True)
        else:
            self.mlp = None
            self.register_buffer('projection_matrix', projection_matrix)
            self.register_buffer("redraw_step", torch.tensor(0))
            self.register_forward_pre_hook(self.__class__._projection_redraw_hook)

    def extra_repr(self) -> str:
        s = ", ".join(
            f"{k}={getattr(self, k)}" for k in ("d", "num_features", "normalize", "uniform_q", "scaling")
        )
        if self.kernel_eps:
            s += f", {self.kernel_eps}"
        return s

    def forward(self, data: Tensor, **kwargs) -> Tensor:
        L, N, H, E = data.shape
        projection = make_orthogonal(self.projection_matrix)

        normalizer = (E ** -0.25) if self.normalize else 1
        data = data * normalizer

        R = projection.shape[-2]
        projection = projection.view(1, *projection.shape).expand(N, -1, -1, -1)
        data = data.movedim(0, -2)
        assert projection.shape == (N, H, E, E)
        assert data.shape == (N, H, L, E)

        proj_data = data.matmul(projection).swapdims(-1, -2)
        assert proj_data.shape == (N, H, E, L)

        out = self.kernel_function(data, proj_data, **kwargs) + self.kernel_eps
        out = out.movedim(-1, 0)
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
        matrices: List[Tensor] = []
        for _ in range(self.num_heads):
            mat = gaussian_orthogonal_random_matrix(
                self.num_features, 
                self.uniform_q, 
                self.scaling, 
                **kwargs
            )
            matrices.append(mat)
        return torch.stack(matrices, 0)

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
        N, H, L, E = data.shape
        projection = projection.swapdims(-1, -2)
        N, H, L, E = projection.shape # (N, R, L)
        E_dim = -1
        L_dim = -2

        normalizer = E ** -0.25
        ratio = (E ** -0.5) # 1 / root(m) normalization from FAVOR

        # h(x) = exp(-||x||**2 / 2)
        h = (
            data
            .pow(2)
            .sum(dim=E_dim, keepdim=True)
            .div(2)
        )

        # additional stabilization
        # NOTE: this seems pretty important, especially for large E / high variance of inputs
        delta = projection - h
        if is_query:
            diff = delta.amax(dim=E_dim, keepdim=True)
        else:
            diff = delta.amax(dim=(E_dim, L_dim), keepdim=True)

        result = ratio * torch.exp(delta - diff)
        return result.swapdims(-1, -2)

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
        delta = projection - h

        if is_query:
            diff = delta.amax(dim=E_dim, keepdim=True)
        else:
            diff = delta.amax(dim=(E_dim, L_dim), keepdim=True)

        result = ratio * torch.exp(delta - diff)
        return result


# non-causal linear attention
def linear_attention(
    q: Tensor, 
    k: Tensor, 
    v: Tensor, 
    fast: bool = True, 
    return_weights: int = 0,
    dropout: float = 0.0,
    stabilizer: float = 1e-6
) -> Tuple[Tensor, Optional[Tensor]]:
    Lq, N, H, Dq = q.shape
    Lk, N, H, Dk = k.shape
    Lv, N, H, Dv = v.shape

    q = q.movedim(0, -2)
    kT = k.movedim(0, -1)
    v = v.movedim(0, -2)

    assert tuple(v.shape) == (N, H, Lv, Dv)
    assert tuple(q.shape) == (N, H, Lq, Dq)
    assert tuple(kT.shape) == (N, H, Dk, Lk)
    weight: Optional[Tensor] = None

    # NOTE: no root d normalization here, its baked into the kernel
    if fast:
        # D_inv = diag(Q'(K_t' @ 1_L))
        # here we express kT @ 1_L as kT.sum(dim=-1))
        # also, diag(a_i, ...).inverse() == diag(1 / a_i, ...)
        orig_dtype = v.dtype
        with torch.cuda.amp.autocast(enabled=False):
            q = q.float()
            kT = kT.float()
            v = v.float()
            D_inv = (
                q.matmul(kT.sum(dim=-1, keepdim=True))
                .view(N, H, Lq, 1)
                .clamp_min(stabilizer)
                .reciprocal()
            )

            out = D_inv * (q @ (kT @ v))
        out = out.to(orig_dtype)

        # normally we wouldn't compute this since it isn't needed to compute performer attn
        if return_weights:
            weight = D_inv[..., :return_weights, :] * (q[..., :return_weights, :] @ kT)

    # Fallback to normal attn (for debugging)
    else:
        q = q / (Dq ** -0.5)
        weight = q.matmul(kT).softmax(dim=-1)
        out = weight.matmul(v)

    out = out.movedim(-2, 0)
    assert out.shape == (Lq, N, H, Dq)
    #assert weight is None or tuple(weight.shape) == (N, H, Lq, Lk)
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
        num_heads: int, 
        dropout: float = 0., 
        bias: bool = True, 
        add_zero_attn: bool = False,
        fast: bool = True,
        kernel_fn: Type[KernelORF] = SoftmaxORF,
        stabilizer: float = 1e-6,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        gini: bool = True,
        **kwargs
    ) -> None:
        super(FAVOR, self).__init__(
            embed_dim, 
            num_heads, 
            dropout, 
            bias, 
            False, 
            add_zero_attn,
            kdim,
            vdim
        )
        self.proj_dim = embed_dim // self.num_heads
        self.fast = fast
        self.stabilizer = stabilizer
        self.kernel = kernel_fn(self.embed_dim, self.num_heads, **kwargs)
        self.gini = gini

        self.buffer_size = 1
        self.register_buffer("last_attn", None)

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
        H = self.num_heads
        E_mh = self.head_dim
        #needs_favor = L >= self.head_dim

        # NOTE: explicitly call MHSA forward to get exact results when not using FAVOR. 
        # MHSA seems to use a C++ implementation and will only agree with pure torch implementation
        # within some (pretty good) level of precision
        if not fast:
            return super().forward(query, key, value, need_weights=need_weights, **kwargs)

        if self._qkv_same_embed_dim:
            Q, K, V = self._in_projection_packed(query, key, value, self.in_proj_weight, self.in_proj_bias)
        else:
            b_q, b_k, b_v = self.in_proj_bias.chunk(3)
            Q, K, V = self._in_projection(query, key, value, self.q_proj_weight, self.k_proj_weight, self.v_proj_weight, b_q, b_k, b_v)

        # multi-head split
        Lq, N, Dq = Q.shape
        Lk, N, Dk = K.shape
        Lv, N, Dv = V.shape
        Q = Q.contiguous().view(Lq, N, H, -1)
        K = K.contiguous().view(Lk, N, H, -1)
        V = V.contiguous().view(Lv, N, H, -1)

        # get kernel function mapping Q -> Q', ...
        Q = self.kernel(Q, is_query=True)
        K = self.kernel(K, is_query=False)

        # attention
        out, weight = linear_attention(Q, K, V, self.fast, self.buffer_size, stabilizer=self.stabilizer)
        out = out.contiguous().view(L, N, -1)

        # gini regularization
        if self.gini:
            gini = self._compute_gini(Q, K, self.stabilizer)
            self._gini = gini

        if weight is not None:
            with torch.no_grad():
                weight = weight.detach().mean(dim=1)
                self.last_attn = weight

        out = self.out_proj(out)
        out = F.dropout(out, self.dropout, training=self.training)
        return out, weight

    @staticmethod
    def favor_layers(model: nn.Module, prefix: str = "") -> Dict[str, "FAVOR"]:
        result: Dict[str, FAVOR] = {}
        if isinstance(model, FAVOR):
            result[prefix] = model
        for name, module in model.named_children():
            p = f"{prefix}.{name}" if prefix else name
            result.update(FAVOR.favor_layers(module, prefix=p))
        return result

    @property
    def last_gini(self) -> Tensor:
        assert self.gini
        return self._gini

    @staticmethod
    def compute_gini(model: nn.Module) -> Dict[str, Tensor]:
        result: Dict[str, Tensor] = {}
        for name, layer in FAVOR.favor_layers(model).items():
            if layer.gini and layer.last_gini is not None:
                result[name] = layer.last_gini.mean()
        return result

    @staticmethod
    def regularizer(model: nn.Module, ord: Union[int, float] = 2, target: Optional[float] = None) -> Tensor:
        layer_gini = FAVOR.compute_gini(model)
        num_layers = len(layer_gini)
        if not num_layers:
            raise RuntimeError(f"No layers had Gini calculation enabled")
        gini = torch.stack([v for v in layer_gini.values()], dim=0)
        if target is not None:
            gini = gini.clamp_min(target)
        gini = torch.linalg.vector_norm(gini, dim=0, ord=ord) / num_layers
        assert gini.numel() == 1
        return gini

    @staticmethod
    def compute_avg_gini(model: nn.Module, ord: Union[int, float] = 2) -> Tensor:
        layer_gini = FAVOR.compute_gini(model)
        num_layers = len(layer_gini)
        if not num_layers:
            raise RuntimeError(f"No layers had Gini calculation enabled")
        gini = torch.stack([v for v in layer_gini.values()], dim=0)
        gini = torch.linalg.vector_norm(gini, dim=0, ord=ord)
        assert gini.numel() == 1
        return gini

    @staticmethod
    def _compute_gini(Q: Tensor, K: Tensor, stabilizer: float = 1e-6) -> Tensor:
        r"""1 - sum(Q @ Kt)"""
        Lq, N, H, Dq = Q.shape
        Lk, N, H, Dk = K.shape

        # We must compute: 
        #   trace[(Q @ Kt) @ (Q @ Kt).T]
        #   = trace[Q @ (Kt @ K) @ Qt]
        #   = trace[Qt @ Q @ Kt @ K]
        #
        # Shapes
        # (L x D) @ (D x L) @ (L x D) @ (D x L)
        # permuted (D x L) @ (L x D) @ (D x L) @ (L x D)
        orig_dtype = Q.dtype
        with torch.cuda.amp.autocast(enabled=False):
            Q = Q.float().movedim(0, -2)
            K = K.float().movedim(0, -2)
            Qt = Q.transpose(-1, -2)
            Kt = K.transpose(-1, -2)

            assert Q.shape == (N, H, Lq, Dq)
            assert K.shape == (N, H, Lk, Dk)
            assert Qt.shape == (N, H, Dq, Lq)
            assert Kt.shape == (N, H, Dk, Lk)

            D_inv = (
                Q.matmul(Kt.sum(dim=-1, keepdim=True))
                .view(N, H, Lq, 1)
                .clamp_min(stabilizer)
                .reciprocal()
            )
            assert D_inv.shape == (N, H, Lq, 1)

            gini = (
                (Qt @ (D_inv * Q) @ Kt @ K)
                .diagonal(dim1=-2, dim2=-1)
                .sum(dim=-1).div(Lk)
            )

        gini = 1 - gini.to(orig_dtype).clamp(min=0, max=1)
        gini = gini.mean(dim=-1)
        assert gini.shape == (N,)
        return gini

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

    @staticmethod
    def _in_projection(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        w_q: Tensor,
        w_k: Tensor,
        w_v: Tensor,
        b_q: Optional[Tensor] = None,
        b_k: Optional[Tensor] = None,
        b_v: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        r"""Taken from F.multi_head_attention_forward"""
        Eq, Ek, Ev = q.size(-1), k.size(-1), v.size(-1)
        assert w_q.shape == (Eq, Eq), f"expecting query weights shape of {(Eq, Eq)}, but got {w_q.shape}"
        assert w_k.shape == (Eq, Ek), f"expecting key weights shape of {(Eq, Ek)}, but got {w_k.shape}"
        assert w_v.shape == (Eq, Ev), f"expecting value weights shape of {(Eq, Ev)}, but got {w_v.shape}"
        assert b_q is None or b_q.shape == (Eq,), f"expecting query bias shape of {(Eq,)}, but got {b_q.shape}"
        assert b_k is None or b_k.shape == (Eq,), f"expecting key bias shape of {(Eq,)}, but got {b_k.shape}"
        assert b_v is None or b_v.shape == (Eq,), f"expecting value bias shape of {(Eq,)}, but got {b_v.shape}"
        return F.linear(q, w_q, b_q), F.linear(k, w_k, b_k), F.linear(v, w_v, b_v)

    @staticmethod
    def set_stabilizer(module: nn.Module, stabilizer: float) -> None:
        if isinstance(module, FAVOR):
            module.stabilizer = stabilizer
        for child in module.children():
            if isinstance(child, FAVOR):
                child.stabilizer = stabilizer
            else:
                FAVOR.set_stabilizer(child, stabilizer)

    @staticmethod
    def disable_favor(module: nn.Module) -> None:
        if isinstance(module, FAVOR):
            module.fast = False
        for child in module.children():
            if isinstance(child, FAVOR):
                child.fast = False
            else:
                FAVOR.disable_favor(child)
                
class PerformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(
        self, 
        d_model: int, 
        nhead: int, 
        dim_feedforward: Optional[int] = None, 
        dropout: float = 0.1, 
        activation: Union[str, nn.Module] = nn.ReLU(),
        feature_redraw_interval: int = 1000,
        fast: bool = True,
        stabilizer: float = 1e-6,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        **kwargs
     ):
        dim_feedforward = dim_feedforward or d_model
        act = "relu" if isinstance(activation, nn.Module) else activation
        super(PerformerEncoderLayer, self).__init__(
            d_model, 
            nhead, 
            dim_feedforward, 
            dropout, 
            act,
            **kwargs
        )
        self.self_attn = FAVOR(
            d_model, 
            nhead, 
            fast=fast, 
            stabilizer=stabilizer, 
            feature_redraw_interval=feature_redraw_interval,
            kernel_eps=0,
            kdim=kdim,
            vdim=vdim,
        )
        if isinstance(activation, nn.Module):
            self.activation = activation

    def duplicate(self) -> "PerformerEncoderLayer":
        new_layer = deepcopy(self)
        new_layer.self_attn = self.self_attn
        new_layer.linear1 = self.linear1
        new_layer.linear2 = self.linear2
        return new_layer

class PerformerDecoderLayer(nn.TransformerDecoderLayer):
    def __init__(
        self, 
        d_model: int, 
        nhead: int, 
        dim_feedforward: Optional[int] = None, 
        dropout: float = 0.1, 
        activation: Union[str, nn.Module] = nn.ReLU(),
        feature_redraw_interval: int = 1000,
        fast: bool = True,
        stabilizer: float = 1e-6,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        **kwargs
     ):
        dim_feedforward = dim_feedforward or d_model
        act = "relu" if isinstance(activation, nn.Module) else activation
        super(PerformerDecoderLayer, self).__init__(
            d_model, 
            nhead, 
            dim_feedforward, 
            dropout, 
            act,
            **kwargs
        )
        self.self_attn = FAVOR(
            d_model, 
            nhead, 
            fast=fast, 
            stabilizer=stabilizer, 
            feature_redraw_interval=feature_redraw_interval,
            kernel_eps=0,
        )
        self.multihead_attn = FAVOR(
            d_model, 
            nhead, 
            fast=fast, 
            stabilizer=stabilizer, 
            feature_redraw_interval=feature_redraw_interval,
            kernel_eps=0,
            kdim=kdim,
            vdim=vdim,
        )
        if isinstance(activation, nn.Module):
            self.activation = activation


class FixedAttention(nn.TransformerDecoderLayer):

    def __init__(
        self, 
        d_model: int, 
        nhead: int, 
        dim_feedforward: Optional[int] = None, 
        dropout: float = 0.0, 
        attn_dropout: float = 0.33, 
        activation: Union[str, nn.Module] = nn.ReLU(),
        track_attn: bool = False
     ):
        dim_feedforward = dim_feedforward or d_model
        act = "relu" if isinstance(activation, nn.Module) else activation
        super().__init__(
            d_model, 
            nhead, 
            dim_feedforward, 
            dropout, 
            activation=act,
        )
        self.track_weights = track_attn
        self.last_attn = self.register_buffer("last_attn", None)
        if isinstance(activation, nn.Module):
            self.activation = activation
        self.multihead_attn.dropout = attn_dropout

    def forward(self, tgt: Tensor, memory: Tensor, **kwargs) -> Tensor:
        tgt2 = self.self_attn(tgt, tgt, tgt)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2, weight = self.multihead_attn(tgt, memory, memory, need_weights=True)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        if weight is not None:
            self.last_attn = weight
        return tgt

    def compute_gini(self) -> Tensor:
        return (1 - self.last_attn.pow(2).sum(dim=-1)).mean()


class PerformerBlock(nn.Module):
    def __init__(self, d: int, mlp_repeats: int, nhead: int, dim_ff: Optional[int] = None, dropout: float = 0.0, activation: nn.Module = nn.ReLU(), drop_path: float = 0.1, **kwargs):
        super().__init__()
        self.d_in = d
        self.nhead = nhead
        self.attn = PerformerEncoderLayer(d, nhead, dim_ff or d, dropout=dropout, activation=activation, **kwargs)
        self.mlps = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.drop_path = DropPath(drop_path)
        for _ in range(mlp_repeats):
            mlp = MLP(d, dim_ff or d, dropout=dropout)
            self.mlps.append(mlp)
            self.norms.append(nn.LayerNorm(d))

    def forward(self, x: Tensor) -> Tensor:
        orig_x = x
        x = self.attn(x)
        for block, norm in zip(self.mlps, self.norms):
            x = norm(x + block(x))

        mask = self.drop_path.get_mask(x)
        return x * mask + orig_x * ~mask

    def duplicate(self) -> "PerformerBlock":
        new_layer = deepcopy(self)
        new_layer.attn = self.attn.duplicate()
        return new_layer



class PerformerDownsample(nn.TransformerDecoder):

    def __init__(
        self, 
        d_in: int,
        d_out: int, 
        nhead: int, 
        num_layers: int = 1,
        dropout: float = 0.1,
        activation: nn.Module = nn.ReLU(),
        **kwargs
     ):
        decoder = PerformerDecoderLayer(d_out, nhead, d_out, dropout=dropout, activation=activation, kdim=d_in, vdim=d_in, **kwargs)
        super().__init__(decoder, num_layers)
        self.d_in = d_in
        self.d_out = d_out
        self.mlp = MLP(d_in, d_out, d_out, dropout=dropout, act=activation)
        self.norm3 = nn.LayerNorm(d_out)

    def forward(self, features: Tensor, keep: Tensor) -> Tensor:
        memory = features
        L, N, D = features.shape
        tgt = features[keep].view(-1, N, D)
        tgt = self.norm3(self.mlp(tgt))
        return super().forward(tgt, memory)


class PerformerUpsample(nn.TransformerDecoder):

    def __init__(
        self, 
        d_in: int,
        d_out: int, 
        nhead: int, 
        num_layers: int = 1,
        dropout: float = 0.1,
        activation: nn.Module = nn.ReLU(),
        **kwargs
     ):
        decoder = PerformerDecoderLayer(d_out, nhead, d_out, dropout=dropout, activation=activation, kdim=d_in, vdim=d_in, **kwargs)
        super().__init__(decoder, num_layers)
        self.d_in = d_in
        self.d_out = d_out

    def forward(self, tgt: Tensor, memory: Tensor) -> Tensor:
        return super().forward(tgt, memory)



class PerformerFPNBlock(nn.Module):

    def __init__(
        self, 
        d: int,
        nhead: int,
        d_low: Optional[int],
        d_high: Optional[int],
        **kwargs
     ):
        super().__init__()
        if d_low is not None:
            self.downsample = PerformerDecoderLayer(d, nhead, d, kdim=d_low, vdim=d_low, **kwargs)
        else:
            self.downsample = None

        if d_high is not None:
            self.upsample = PerformerDecoderLayer(d, nhead, d, kdim=d_high, vdim=d_high, **kwargs)
        else:
            self.upsample = None

    def forward(self, features: Tensor, down: Optional[Tensor], up: Optional[Tensor]) -> Tensor:
        if down is not None and self.downsample is not None:
            features = self.downsample(features, down)
        if up is not None and self.upsample is not None:
            features = self.upsample(features, up)
        return features


class PerformerFPN(nn.Module):

    def __init__(
        self, 
        d: List[int],
        nhead: int,
        dropout: float = 0,
        drop_path: float = 0.1,
        **kwargs
     ):
        super().__init__()
        self.drop_path = DropPath(drop_path)
        self.blocks = nn.ModuleList()
        for i, _d in enumerate(d):
            if i == 0:
                d_low = None
            else:
                d_low = d[i-1]

            if i == len(d) - 1:
                d_high = None
            else:
                d_high = d[i+1]

            layer = PerformerFPNBlock(_d, nhead, d_low, d_high, dropout=dropout, **kwargs)
            self.blocks.append(layer)

    def forward(self, features: List[Tensor]) -> List[Tensor]:
        assert len(features) == len(self.blocks)
        out_features = []
        for i, (feature, block) in enumerate(zip(features, self.blocks)):
            if i == 0:
                down = None
            else:
                down = features[i-1]

            if i == len(features) - 1:
                up = None
            else:
                up = features[i + 1]
            feature = block(feature, down, up)
            mask = self.drop_path.get_mask(feature)
            feature = mask * feature + (~mask) * features[i]
            out_features.append(feature)
        return out_features


class VisionPerformer(nn.Module):
    def __init__(self, blocks: List[PerformerBlock], repeats: List[int], fpn_repeats: int = 0, act: nn.Module = nn.Mish(), share_weights: bool = False):
        super().__init__()

        self.levels = nn.ModuleList()
        self.downs = nn.ModuleList()
        for i, (r, b) in enumerate(zip(repeats, blocks)):
            level = []
            for _ in range(r):
                if share_weights:
                    level.append(b.duplicate())
                else:
                    level.append(deepcopy(b))
            self.levels.append(nn.ModuleList(level))

            d_next = blocks[i+1].d_in if i+1 < len(blocks) else None
            if d_next is None:
                continue
            down = PerformerDownsample(b.d_in, d_next, b.nhead, activation=act)
            self.downs.append(down)

        self.bifpn = nn.ModuleList()
        for _ in range(fpn_repeats):
            d = [x.d_in for x in blocks] + [blocks[-1].d_in]
            layer = PerformerFPN(d, nhead=1, activation=act)
            self.bifpn.append(layer)

    def forward(self, features: Tensor, *args, **kwargs) -> List[Tensor]:
        N = features.shape[1]
        H, W = kwargs["H"], kwargs["W"]
        pos_enc = kwargs.get("pos_enc", None)
        fpn = []
        for i, l in enumerate(self.levels):
            for block in l:
                #if pos_enc is not None:
                #    features = features + pos_enc #+ F.dropout(pos_enc, 0.1, training=self.training)
                features = block(features)
            fpn.append(features)
            if i < len(self.downs):
                keep = self.downsample(features, H=H, W=W)
                H //= 2
                W //= 2
                features = self.downs[i](features, keep)
        fpn.append(features)

        for layer in self.bifpn:
            fpn = layer(fpn)

        return fpn

    def downsample(self, features: Tensor, *args, **kwargs) -> Tensor:
        L, N, D = features.shape
        H = kwargs["H"]
        W = kwargs["W"]
        keep = features.new_zeros(L, N, dtype=torch.bool)
        keep.view(H, W, N)[::2, ::2, ...] = True
        return keep


class PointPerformer(VisionPerformer):

    def forward(self, features: Tensor, *args, **kwargs) -> List[Tensor]:
        N = features.shape[1]
        fpn = []
        for i, l in enumerate(self.levels):
            for block in l:
                features = block(features)
            fpn.append(features)
            if i < len(self.downs):
                keep = self.downsample(features, *args, **kwargs)
                features = self.downs[i](features, keep)
        fpn.append(features)

        for layer in self.bifpn:
            fpn = layer(fpn)

        return fpn

    def downsample(self, features: Tensor, *args, **kwargs) -> Tensor:
        L, N, D = features.shape
        keep = torch.randperm(L)[:L//4]
        return keep

    @staticmethod
    def upsample(features: Tensor, coords: Tensor, original_coords: Tensor) -> Tensor:
        try:
            import torch_cluster as tc
        except Exception:
            print(f"PointPerformer.upsample requires torch-cluster")
            raise
        L, N, D = features.shape
        L, N, C = coords.shape
        L2, N, C2 = original_coords.shape
        assert C == C2
        batch_coords = torch.arange(N, device=features.device).expand(L, -1).view(-1)
        batch_orig_coords = torch.arange(N, device=features.device).expand(L2, -1).view(-1)
        clusters = tc.nearest(original_coords.view(-1, C), coords.view(-1, C), batch_orig_coords, batch_coords)
        return features.view(-1, D)[clusters].view(-1, N, D).contiguous()


class PartialTrainingMixin:
    original_trainable: Dict[str, bool] = {}

    def freeze_params(self, p: float) -> None:
        if not self.original_trainable:
            self.original_trainable = {name: param.requires_grad for name, param in self.named_parameters()}

        for param in self.parameters():
            trainable = random.random() >= p
            param.requires_grad = trainable


    def unfreeze_params(self) -> None:
        for name, param in self.original_trainable.parameters():
            if self.original_trainable[name]:
                param.requires_grad = True
            else:
                param.requires_grad = False
