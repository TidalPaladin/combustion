#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
from torch import Tensor
from typing import Any, Tuple, List, Optional
from copy import deepcopy
import torch.nn as nn

#from .performer import PerformerLayer, FAVOR
from .position import RelativeLearnableFourierFeatures, RelativePositionalEncoder
from ..cluster import TransitionDown, TransitionUp, KNNCluster, FarthestPointsDecimate
from .common import MLP, SqueezeExcite

class CrossAttention(nn.Module):

    def __init__(self, d: int, nhead: int, dim_ff: Optional[int] = None, dropout: float = 0.1, act: nn.Module = nn.Mish()):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, nhead)
        self.norm1 = nn.LayerNorm(d)

        dim_ff = dim_ff or d
        self.mlp = MLP(d, dim_ff, dropout=dropout, act=act)
        self.norm2 = nn.LayerNorm(d)

    def forward(self, tgt: Tensor, src: Tensor) -> Tensor:
        attn = self.attn(tgt, src, src)[0]
        tgt = self.norm1(tgt + attn)
        tgt = self.norm2(tgt + self.mlp(tgt))
        return tgt

    def duplicate(self) -> "CrossAttention":
        new = deepcopy(self)
        new.attn = self.attn
        new.mlp = self.mlp
        return new

class KNNTail(nn.Module):

    def __init__(self, d_in: int, d_out: int, k: int, nhead: int = 1, repeats: int = 1, dropout: float = 0, pos_features: int = 128, **kwargs):
        super().__init__()
        self.mlp = MLP(d_in, d_out, d_out, dropout=dropout)
        self.se1 = SqueezeExcite(d_out, d_out // 2)
        self.norm_mlp = nn.LayerNorm(d_out)
        self.norm_se = nn.LayerNorm(d_out)
        self.pos_enc = RelativeLearnableFourierFeatures(3, pos_features, d_out)

        self.cluster = KNNCluster(k)
        self.norm_mean = nn.LayerNorm(d_out)

        self.mixer = nn.ModuleList([CrossAttention(d_out, nhead, d_out*4, dropout=dropout)]*repeats)

        self.se2 = SqueezeExcite(d_out, d_out // 2)
        self.se_norm = nn.LayerNorm(d_out)

        self.final_norm = nn.LayerNorm(d_out)

    def forward(self, coords: Tensor, features: Optional[Tensor]=None) -> Tensor:
        L, N, C = coords.shape
        attn_idx = self.cluster(coords, coords)
        pos_emb = self.pos_enc(coords, coords[attn_idx].view(-1, L, N, C))

        if features is None:
            features = coords
        assert features is not None

        features = self.norm_mlp(self.mlp(features))
        features = self.norm_se(features + self.se1(features))
        L, N, D = features.shape

        orig_features = features
        f = features[attn_idx].view(-1, L, N, D)
        f += pos_emb
        tgt = features.view(-1, L*N, D)
        src = f.view(-1, L * N, D)
        for layer in self.mixer:
            tgt = layer(tgt, src)
        features = tgt.view(L, N, D)
        #features = self.se_norm(features + self.se_norm(features))
        #features = f.view(-1, L, N, D)
        #features = self.final_norm(orig_features + features.mean(dim=0))
        assert features.shape == (L, N, D)

        return features


class KNNDownsample(nn.Module):

    def __init__(self, d: int, d_out: int, ratio: float = 0.25, **kwargs):
        super().__init__()
        self.down = FarthestPointsDecimate(ratio)
        self.mlp = MLP(d, d_out, d_out, **kwargs)
        self.norm = nn.LayerNorm(d_out)
        self.register_buffer("coords", None)

    def forward(self, coords: Tensor, features: Tensor) -> Tuple[Tensor, Tensor]:
        L1, N, D1 = features.shape
        C = coords.shape[-1]

        keep_coords = self.down(coords)
        features = features[keep_coords].view(-1, N, D1)
        coords = coords[keep_coords].view(-1, N, C)

        features = self.norm(self.mlp(features))
        return coords, features



class PCTDown(nn.Module):

    def __init__(self, d: int, k: int, pos_enc: RelativePositionalEncoder, repeats: int = 1, ratio: float = 0.25, **kwargs):
        super().__init__()
        self.pos_enc = pos_enc
        layer = nn.TransformerEncoderLayer(2*d, **kwargs)
        self.attn = nn.TransformerEncoder(layer, repeats)
        self.down = TransitionDown(d, 2*d, k, ratio, act=nn.SiLU())
        self.cluster = KNNCluster(k)
        self.register_buffer("coords", None)

    def forward(self, coords: Tensor, features: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        L1, N, D1 = features.shape
        C = coords.shape[-1]

        keep_coords, pool_features, neighbor_idx, keep_idx = self.down(coords, features)
        L2, N, D2 = pool_features.shape

        attn_idx = self.cluster(keep_coords, keep_coords)

        pos_emb = self.pos_enc(keep_coords, keep_coords[attn_idx].view(-1, L2, N, C))
        pool_features = pool_features[attn_idx].view(-1, L2, N, D2) + pos_emb

        pool_features = self.attn(pool_features.view(-1, L2 * N, D2)).view(-1, L2, N, D2)
        pool_features = pool_features.mean(dim=0)

        setattr(self, "coords", keep_coords.clone())
        assert self.coords is not None
        return keep_coords, pool_features, neighbor_idx, keep_idx, attn_idx

class PCTUp(nn.Module):

    def __init__(self, d: int, k: int, pos_enc: RelativePositionalEncoder, repeats: int = 1, **kwargs):
        super().__init__()
        self.pos_enc = pos_enc
        layer = nn.TransformerEncoderLayer(d, **kwargs)
        self.attn = nn.TransformerEncoder(layer, repeats)
        self.up = TransitionUp(2*d, d, act=nn.SiLU())
        self.cluster = KNNCluster(k)
        self.register_buffer("coords", None)

    def forward(self, features_coarse: Tensor, features_fine: Tensor, neighbor_idx: List[Tensor], keep_idx: List[Tensor], coords: Tensor, attn_idx: Optional[Tensor]) -> Tensor:
        features_fine = self.up(features_coarse, features_fine, neighbor_idx, keep_idx)
        L, N, D = features_fine.shape
        C = coords.shape[-1]

        if attn_idx is None:
            attn_idx = self.cluster(coords, coords)

        features_fine = features_fine[attn_idx].view(-1, L, N, D)
        pos_emb = self.pos_enc(coords, coords[attn_idx].view(-1, L, N, C))
        features_fine = features_fine + pos_emb

        features_fine = self.attn(features_fine.view(-1, L*N, D)).view(-1, L, N, D)
        features_fine = features_fine.mean(dim=0)
        setattr(self, "coords", coords.clone())
        assert self.coords is not None
        return features_fine



class ClusterModel(nn.Module):

    def __init__(self, d: int, d_in: int, num_coords: int = 3, blocks: List[int] = [1, 1, 1, 1], k: int = 32, **kwargs):
        super().__init__()
        self.tail = MLP(d_in, d, d)
        self.first_mlp = MLP(d, d, d)

        #self.initial_decimate = InitialTransitionDown(d, d, max_points=32000)

        encoder = []
        decoder = []
        for i, repeats in enumerate(blocks):
            d_block = int(d * 2 ** i)
            pos_enc = RelativePositionalEncoder(num_coords, 2*d_block)
            block = PCTDown(d_block, k, pos_enc, dim_feedforward=4*d_block, **kwargs)
            encoder.append(block)
            pos_enc = RelativePositionalEncoder(num_coords, d_block)
            block = PCTUp(d_block, k, pos_enc, dim_feedforward=2*d_block, **kwargs)
            decoder.append(block)

        self.encoder = nn.ModuleList(encoder)
        self.decoder = nn.ModuleList(reversed(decoder))

        max_d = d * 2 ** len(blocks)
        self.bottom = MLP(max_d, max_d, max_d)

        # first block
        self.pos_enc = RelativePositionalEncoder(num_coords, d)
        layer = nn.TransformerEncoderLayer(d, **kwargs)
        self.attn = nn.TransformerEncoder(layer, 1)
        self.cluster = KNNCluster(k)


    def forward(self, coords: Tensor, features: Tensor) -> Tensor:
        features = self.tail(features)

        L, N, D = features.shape
        C = coords.shape[-1]

        first_attn_idx = self.cluster(coords, coords)
        features = features[first_attn_idx].view(-1, L, N, D)
        pos_emb = self.pos_enc(coords, coords[first_attn_idx].view(-1, L, N, C))
        features = features + pos_emb
        features = self.attn(features.view(-1, L*N, D)).view(-1, L, N, D)
        features = features.mean(dim=0)
        first_features = self.first_mlp(features)

        skip_conns: List[List[Tensor]] = []

        C = coords.shape[-1]
        N = coords.shape[-2]

        for i, enc in enumerate(self.encoder):
            keep_coords, coarse_features, neighbor_idx, keep_idx, attn_idx = enc(coords, features)
            skip_conns.append(
                [coords, features, neighbor_idx, keep_idx, attn_idx]
            )
            features = coarse_features
            coords = keep_coords

        features = self.bottom(features)

        skip_conns = list(reversed(skip_conns))
        for i, dec in enumerate(self.decoder):
            coords, fine_features, neighbor_idx, keep_idx, _ = skip_conns[i]
            attn_idx = skip_conns[i+1][-1] if i < len(skip_conns) - 1 else first_attn_idx
            features = dec(features, fine_features, neighbor_idx, keep_idx, coords, attn_idx)

        features = features + first_features
        return features

#class PerformerDownsample(nn.Module):
#
#    def __init__(
#        self, 
#        d_in: int,
#        d_out: int, 
#        nhead: int, 
#        dropout: float = 0.1, 
#        activation: nn.Module = nn.ReLU(),
#        feature_redraw_interval: int = 1000,
#        fast: bool = True,
#        stabilizer: float = 1e-6,
#        kdim: Optional[int] = None,
#        vdim: Optional[int] = None,
#     ):
#        super().__init__()
#        self.attn = FAVOR(
#            d_out, 
#            nhead, 
#            fast=fast, 
#            stabilizer=stabilizer, 
#            feature_redraw_interval=feature_redraw_interval,
#            kdim=kdim,
#            vdim=vdim,
#        )
#        self.norm1 = nn.LayerNorm(d_out)
#        self.mlp = MLP(d_in, d_out, d_out, dropout=dropout, act=activation)
#        self.norm2 = nn.LayerNorm(d_out)
#
#    def forward(self, features: Tensor, keep: Tensor) -> Tensor:
#        L, N, D = features.shape
#        Q = features[keep].view(-1, N, D)
#        K = V = features
#
#        features = self.norm1(Q + self.attn(Q, K, V)[0])
#        features = self.norm2(features + self.mlp(features))
#        return features
