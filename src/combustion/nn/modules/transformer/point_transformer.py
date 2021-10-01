#!/usr/bin/env python
# -*- coding: utf-8 -*-
from copy import deepcopy
from typing import List, Optional, Tuple

import torch.nn as nn
from torch import Tensor

from ....util import MISSING
from ..cluster import FarthestPointsDecimate, Indices, KNNCluster, NearestCluster, TransitionDown, TransitionUp
from .common import MLP

# from .performer import PerformerLayer, FAVOR
from .position import RelativePositionalEncoder


class CrossAttention(nn.Module):
    def __init__(
        self, d: int, nhead: int, dim_ff: Optional[int] = None, dropout: float = 0.1, act: nn.Module = nn.Mish()
    ):
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
    def __init__(
        self,
        d_out: int,
        nhead: int = 1,
        repeats: int = 1,
        dropout: float = 0,
        **kwargs,
    ):
        super().__init__()
        # self.pos_enc = RelativePositionalEncoder(3, d_out, dropout=dropout, act=nn.Mish())
        self.d_out = d_out
        self.mixer = nn.ModuleList([CrossAttention(d_out, nhead, d_out * 4, dropout=dropout) for _ in range(repeats)])

    def forward(self, features: Tensor, indices: Indices) -> Tensor:
        # Lc, Nc, C = coords.shape
        Lf, Nf, D = features.shape
        # assert Lc == Lf
        # assert Nc == Nf
        L = Lf
        N = Nf

        # build neighborhoods
        # rel_pos_emb = self.pos_enc(coords, indices.apply_knn(coords))

        # attention between query point and it's neighborhood
        for layer in self.mixer:
            rel_features = indices.apply_knn(features)
            # Kp, _, _, _ = rel_pos_emb.shape
            Kf, _, _, _ = rel_features.shape
            # assert Kp == Kf
            # rel_features += rel_pos_emb

            tgt = features.view(-1, L * N, D)
            src = rel_features.view(-1, L * N, D)
            tgt = layer(tgt, src)

            features = tgt.view(L, N, D)

        return features


class KNNDownsample(nn.Module):
    def __init__(self, d: int, d_out: int, **kwargs):
        super().__init__()
        self.mlp = MLP(d, d_out, d_out, **kwargs)
        self.norm = nn.LayerNorm(d_out)

    def forward(self, features: Tensor, indices: Indices) -> Tensor:
        L1, N, D1 = features.shape
        features = indices.apply_downsample(features)
        features = self.norm(self.mlp(features))
        return features


class KNNUpsample(nn.Module):
    def __init__(self, d: int, d_out: int, **kwargs):
        super().__init__()
        self.mlp = MLP(d, d_out, d_out, **kwargs)
        self.norm = nn.LayerNorm(d_out)

    def forward(self, down_features: Tensor, up_features: Tensor, indices: Indices) -> Tensor:
        assert indices.upsample is not MISSING
        Ld, N, Dd = down_features.shape
        Lu, N, Du = up_features.shape

        # match down channels with up channels
        down_features = self.mlp(down_features)
        assert down_features.shape[-1] == Du

        # upsample + add
        up_features = up_features + indices.apply_upsample(down_features)

        return up_features


class KNNEncoder(nn.Module):
    def __init__(
        self,
        d_in: int,
        repeats: List[int],
        nhead: int = 1,
        dropout: float = 0,
        upsample: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.attention = nn.ModuleList()
        self.down = nn.ModuleList()
        self.upsample = upsample

        d = d_in
        for r in repeats:
            attn = KNNTail(d, nhead, r, dropout=dropout, **kwargs)
            down = KNNDownsample(d, 2 * d, **kwargs)
            self.attention.append(attn)
            self.down.append(down)
            d *= 2

    @property
    def num_levels(self) -> int:
        return len(self.attention)

    @property
    def num_channels(self) -> List[int]:
        return [x.d_out for x in self.attention]

    def forward(self, features: Tensor, indices: List[Indices]) -> List[Tensor]:
        assert len(indices) == self.num_levels
        out_features: List[Tensor] = []
        for idx, attn, down in zip(indices, self.attention, self.down):
            features = attn(features, idx)
            out_features.append(features)
            features = down(features, idx)
        out_features.append(features)
        return out_features


class KNNDecoder(nn.Module):
    def __init__(
        self,
        d_in: int,
        repeats: List[int],
        nhead: int = 1,
        dropout: float = 0,
        **kwargs,
    ):
        super().__init__()
        self.attention = nn.ModuleList()
        self.up = nn.ModuleList()

        d = d_in
        for r in repeats:
            up = KNNUpsample(2 * d, d, **kwargs)
            attn = KNNTail(d, nhead, r, dropout=dropout, **kwargs)
            self.up.append(up)
            self.attention.append(attn)
            d *= 2

    @property
    def num_levels(self) -> int:
        return len(self.attention)

    def forward(self, features: List[Tensor], indices: List[Indices]) -> List[Tensor]:
        for i in range(self.num_levels):
            upsample = self.up[-(i + 1)]
            attn = self.attention[-(i + 1)]
            idx = indices[-(i + 1)]
            down_features = features[-(i + 1)]
            up_features = features[-(i + 2)]

            up_features = upsample(down_features, up_features, idx)
            up_features = attn(up_features, idx)
            features[-(i + 2)] = up_features

        return features


class PointTransformer(nn.Module):
    def __init__(self, d: int, repeats: List[int], nhead: int = 1, dropout: float = 0, act: nn.Module = nn.Mish()):
        super().__init__()
        self.encoder = KNNEncoder(d, repeats, nhead, dropout, act=act)
        # d_lowest = self.encoder.num_channels[-1] * 2
        # self.lowest = KNNTail(d_lowest, nhead, 3, dropout, act=act)
        self.decoder = KNNDecoder(d, repeats, nhead, dropout, act=act)

    def get_indices(self, coords: Tensor, k: int, ratio: float = 0.25) -> List[Indices]:
        knn = KNNCluster(k)
        down = FarthestPointsDecimate(ratio)
        up = NearestCluster()
        indices: List[Indices] = []

        for _ in range(self.encoder.num_levels):
            idx = Indices.create(coords, knn, down, up)
            indices.append(idx)
            coords = idx.apply_downsample(coords)

        return indices

    def forward(self, features: Tensor, indices: List[Indices]) -> Tensor:
        fpn = self.encoder(features, indices)
        # fpn[-1] = self.lowest(fpn[-1])
        fpn = self.decoder(fpn, indices)
        return fpn[0]


class PCTDown(nn.Module):
    def __init__(
        self, d: int, k: int, pos_enc: RelativePositionalEncoder, repeats: int = 1, ratio: float = 0.25, **kwargs
    ):
        super().__init__()
        self.pos_enc = pos_enc
        layer = nn.TransformerEncoderLayer(2 * d, **kwargs)
        self.attn = nn.TransformerEncoder(layer, repeats)
        self.down = TransitionDown(d, 2 * d, k, ratio, act=nn.SiLU())
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
        self.up = TransitionUp(2 * d, d, act=nn.SiLU())
        self.cluster = KNNCluster(k)
        self.register_buffer("coords", None)

    def forward(
        self,
        features_coarse: Tensor,
        features_fine: Tensor,
        neighbor_idx: List[Tensor],
        keep_idx: List[Tensor],
        coords: Tensor,
        attn_idx: Optional[Tensor],
    ) -> Tensor:
        features_fine = self.up(features_coarse, features_fine, neighbor_idx, keep_idx)
        L, N, D = features_fine.shape
        C = coords.shape[-1]

        if attn_idx is None:
            attn_idx = self.cluster(coords, coords)

        features_fine = features_fine[attn_idx].view(-1, L, N, D)
        pos_emb = self.pos_enc(coords, coords[attn_idx].view(-1, L, N, C))
        features_fine = features_fine + pos_emb

        features_fine = self.attn(features_fine.view(-1, L * N, D)).view(-1, L, N, D)
        features_fine = features_fine.mean(dim=0)
        setattr(self, "coords", coords.clone())
        assert self.coords is not None
        return features_fine


class ClusterModel(nn.Module):
    def __init__(self, d: int, d_in: int, num_coords: int = 3, blocks: List[int] = [1, 1, 1, 1], k: int = 32, **kwargs):
        super().__init__()
        self.tail = MLP(d_in, d, d)
        self.first_mlp = MLP(d, d, d)

        # self.initial_decimate = InitialTransitionDown(d, d, max_points=32000)

        encoder = []
        decoder = []
        for i, repeats in enumerate(blocks):
            d_block = int(d * 2 ** i)
            pos_enc = RelativePositionalEncoder(num_coords, 2 * d_block)
            block = PCTDown(d_block, k, pos_enc, dim_feedforward=4 * d_block, **kwargs)
            encoder.append(block)
            pos_enc = RelativePositionalEncoder(num_coords, d_block)
            block = PCTUp(d_block, k, pos_enc, dim_feedforward=2 * d_block, **kwargs)
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
        features = self.attn(features.view(-1, L * N, D)).view(-1, L, N, D)
        features = features.mean(dim=0)
        first_features = self.first_mlp(features)

        skip_conns: List[List[Tensor]] = []

        C = coords.shape[-1]
        N = coords.shape[-2]

        for i, enc in enumerate(self.encoder):
            keep_coords, coarse_features, neighbor_idx, keep_idx, attn_idx = enc(coords, features)
            skip_conns.append([coords, features, neighbor_idx, keep_idx, attn_idx])
            features = coarse_features
            coords = keep_coords

        features = self.bottom(features)

        skip_conns = list(reversed(skip_conns))
        for i, dec in enumerate(self.decoder):
            coords, fine_features, neighbor_idx, keep_idx, _ = skip_conns[i]
            attn_idx = skip_conns[i + 1][-1] if i < len(skip_conns) - 1 else first_attn_idx
            features = dec(features, fine_features, neighbor_idx, keep_idx, coords, attn_idx)

        features = features + first_features
        return features


# class PerformerDownsample(nn.Module):
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
