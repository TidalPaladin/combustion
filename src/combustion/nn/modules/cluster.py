#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from ...util import MISSING


try:
    import torch_cluster as tc
except ImportError:
    tc: Any = None

try:
    import torch_scatter as ts
except ImportError:
    ts: Any = None


def flatten_batch(x: Tensor) -> Tuple[Tensor, Tensor]:
    N, L, C = x.shape
    batch_idx = torch.arange(N, device=x.device).view(N, 1).expand(N, L).contiguous().view(-1)
    x = x.contiguous().view(-1, C).contiguous()
    return x, batch_idx


class Decimate(nn.Module, ABC):
    def __init__(self, ratio: float = 0.25, max_points: Optional[int] = None):
        super().__init__()
        self.ratio = ratio
        self.max_points = max_points

    def extra_repr(self) -> str:
        s = f"ratio={self.ratio}"
        if self.max_points is not None:
            s += f"max_points={self.max_points}"
        return s

    @abstractmethod
    def forward(self, coords: Tensor) -> List[Tensor]:
        if tc is None:
            raise ImportError(f"{self.__class__.__name__} requires torch-cluster")
        else:
            assert tc is not None


class Cluster(nn.Module, ABC):
    def __init__(self, k: int):
        super().__init__()
        self.k = k

    def extra_repr(self) -> str:
        s = f"k={self.k}"
        return s

    @abstractmethod
    def forward(self, coords1: Tensor, coords2: Tensor) -> List[Tensor]:
        if tc is None:
            raise ImportError(f"{self.__class__.__name__} requires torch-cluster")
        else:
            assert tc is not None


class RandomDecimate(Decimate):
    def __init__(self, ratio: float = 0.25, max_points: Optional[int] = None, seed: int = 42):
        super().__init__(ratio, max_points)
        self.seed = seed

    def forward(self, coords: Tensor) -> List[Tensor]:
        super().forward(coords)

        L, N, C = coords.shape
        K = int(L * self.ratio)

        with torch.no_grad():
            with torch.random.fork_rng(devices=[coords.device]):
                torch.random.manual_seed(self.seed)
                keep = torch.randperm(L, device=coords.device)[:K]
                keep = torch.meshgrid(keep, torch.arange(N, device=coords.device))
                keep = [x.contiguous().view(-1) for x in keep]

        return keep


class FarthestPointsDecimate(Decimate):
    def __init__(self, ratio: float = 0.25, max_points: Optional[int] = None):
        super().__init__(ratio, max_points)

    def forward(self, coords: Tensor) -> List[Tensor]:
        super().forward(coords)
        L, N, C = coords.shape

        self.ratio
        if self.max_points is not None:
            self.max_points / L

        with torch.no_grad():
            # torch cluster requires batch_idx to be sorted, so permute before flattening
            coords = coords.swapdims(0, 1)
            coords, batch_idx = flatten_batch(coords)

            # farthest point sampling to find points to keep
            keep = tc.fps(coords, batch_idx, self.ratio)
            keep = keep.view(N, -1).swapdims(0, 1).fmod_(L)
            K = keep.shape[0]

            indices = torch.meshgrid(torch.arange(K, device=coords.device), torch.arange(N, device=coords.device))
            indices = [keep.contiguous().view(-1), indices[-1].contiguous().view(-1)]

        return indices


class KNNCluster(Cluster):
    def __init__(self, k: int, cosine: bool = False, num_workers: int = 8, cpu_threshold=4096):
        super().__init__(k)
        self.cosine = cosine
        self.num_workers = num_workers

    def extra_repr(self) -> str:
        s = f"k={self.k}"
        if self.cosine:
            s += f", cosine=True"
        return s

    def forward(self, coords1: Tensor, coords2: Tensor) -> List[Tensor]:
        if tc is None:
            raise ImportError(f"{self.__class__.__name__} requires torch-cluster")
        else:
            assert tc is not None

        L1, N, C = coords1.shape
        L2, N, C = coords2.shape
        K = self.k
        graph_knn = coords1 is coords2

        with torch.no_grad():
            coords1 = coords1.swapdims(0, 1)
            coords1, batch_idx1 = flatten_batch(coords1)

            if graph_knn:
                coords2 = coords1
                batch_idx2 = batch_idx1
            else:
                coords2 = coords2.swapdims(0, 1)
                coords2, batch_idx2 = flatten_batch(coords2)

            _, clusters = tc.knn(coords1, coords2, self.k, batch_idx1, batch_idx2, cosine=self.cosine)

            # turn clusters into a set of indices into `features` or `coords`
            clusters = clusters.view(N, L2, K).permute(-1, 1, 0).fmod_(L2)
            assert tuple(clusters.shape) == (K, L2, N)
            indices = torch.meshgrid(
                torch.arange(K, device=coords2.device),
                torch.arange(L2, device=coords2.device),
                torch.arange(N, device=coords2.device),
            )
            indices = [clusters.contiguous().view(-1), indices[-1].contiguous().view(-1)]
            assert len(indices[0]) == L2 * N * K

        return indices


class NearestCluster(Cluster):
    def __init__(self, num_workers: int = 8, cpu_threshold=4096):
        super().__init__(k=1)
        self.num_workers = num_workers

    def forward(self, coords1: Tensor, coords2: Tensor) -> List[Tensor]:
        if tc is None:
            raise ImportError(f"{self.__class__.__name__} requires torch-cluster")
        else:
            assert tc is not None

        L1, N, C = coords1.shape
        L2, N, C = coords2.shape
        self.k
        graph_knn = coords1 is coords2

        with torch.no_grad():
            coords1 = coords1.swapdims(0, 1)
            coords1, batch_idx1 = flatten_batch(coords1)

            if graph_knn:
                coords2 = coords1
                batch_idx2 = batch_idx1
            else:
                coords2 = coords2.swapdims(0, 1)
                coords2, batch_idx2 = flatten_batch(coords2)

            clusters = tc.nearest(coords2, coords1, batch_idx2, batch_idx1)

            # turn clusters into a set of indices into `features` or `coords`
            clusters = clusters.view(N, L2).T.fmod_(L1)
            assert tuple(clusters.shape) == (L2, N)
            indices = torch.meshgrid(
                torch.arange(L2, device=coords2.device),
                torch.arange(N, device=coords2.device),
            )
            indices = [clusters.contiguous().view(-1), indices[-1].contiguous().view(-1)]
            assert len(indices[0]) == L2 * N

        return indices


@dataclass
class Indices:
    knn: List[Tensor] = MISSING
    downsample: List[Tensor] = MISSING
    upsample: List[Tensor] = MISSING

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}("
        for name in self.__dataclass_fields__.keys():
            field = getattr(self, name)
            if isinstance(field, list) and len(field) and isinstance(field[0], Tensor):
                field_repr = self._tensor_list_repr(field)
            else:
                field_repr = repr(field)
            s += f"{name}={field_repr}, "
        s = s[:-2]
        s += ")"
        return s

    def apply_knn(self, x: Tensor) -> Tensor:
        self._check_attr("knn")
        L, N, D = x.shape
        x = x[self.knn].view(-1, L, N, D).contiguous()
        return x

    def apply_downsample(self, x: Tensor) -> Tensor:
        self._check_attr("downsample")
        L, N, D = x.shape
        x = x[self.downsample].view(-1, N, D).contiguous()
        return x

    def apply_upsample(self, x: Tensor) -> Tensor:
        self._check_attr("upsample")
        L, N, D = x.shape
        x = x[self.upsample].view(-1, N, D).contiguous()
        return x

    def _check_attr(self, attr: str) -> None:
        attr_val = getattr(self, attr)
        if attr_val is MISSING:
            raise AttributeError(f"{attr} is MISSING")
        if len(attr_val) != 2:
            raise ValueError(f"Expected {attr} to be a length-2 list, found length-{len(attr_val)}")

    @staticmethod
    def _tensor_list_repr(l: List[Tensor]) -> str:
        num_items = len(l)
        item_size = l[0].numel()
        return f"({num_items}, {item_size})"

    @classmethod
    def create(
        cls,
        coords: Tensor,
        knn: Optional[KNNCluster] = None,
        down: Optional[Decimate] = None,
        up: Optional[NearestCluster] = None,
    ) -> "Indices":
        knn_idx = knn(coords, coords) if knn is not None else MISSING
        down_idx = down(coords) if down is not None else MISSING
        if down is not None and up is not None:
            indices = cls(knn=knn_idx, downsample=down_idx)
            down_coords = indices.apply_downsample(coords)
            nearest_idx = up(down_coords, coords)
            indices.upsample = nearest_idx
        else:
            indices = cls(knn=knn_idx, downsample=down_idx)
        return indices

    @staticmethod
    def unbatch_indices(idx: Tensor, dim: int = 1) -> List[Tensor]:
        assert idx.ndim == 2
        N = idx.shape[dim]

        batch = torch.arange(N, dtype=idx.dtype, device=idx.device).unsqueeze(0).movedim(-1, dim).expand_as(idx)
        idx = idx.flatten()
        batch = batch.flatten()

        result = [idx, idx]
        result[dim] = batch
        return result


class MLP(nn.Module):
    def __init__(self, d_in: int, d_ff: int, d_out: int, act: nn.Module = nn.ReLU()):
        super().__init__()
        self.linear1 = nn.Linear(d_in, d_ff, bias=False)
        self.norm1 = nn.LayerNorm(d_ff)
        self.linear2 = nn.Linear(d_ff, d_out, bias=False)
        self.norm2 = nn.LayerNorm(d_out)
        self.act = act

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear1(x)
        x = self.norm1(x)
        x = self.linear2(x)
        x = self.norm2(x)
        return self.act(x)


class TransitionDown(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        k: int,
        ratio: float = 0.25,
        act: nn.Module = nn.ReLU(),
        max_points: Optional[int] = None,
    ):
        super().__init__()
        # self.decimate =  RandomDecimate(ratio)
        self.decimate = RandomDecimate(ratio, max_points=max_points)
        self.cluster = KNNCluster(k)
        self.mlp = MLP(dim, dim_out, dim_out, act=act)

    def forward(self, coords: Tensor, features: Tensor) -> Tuple[Tensor, Tensor, List[Tensor], List[Tensor]]:
        L, N, D = features.shape
        C = coords.shape[-1]
        features = self.mlp(features)
        D2 = features.shape[-1]

        keep_idx = self.decimate(coords)
        keep_coords = coords[keep_idx].view(-1, N, C)
        L2 = keep_coords.shape[0]

        neighbor_idx = self.cluster(coords, keep_coords)
        pool_features = features[neighbor_idx].view(self.k, L2, N, D2)
        pool_features = pool_features.amax(dim=0)

        return keep_coords, pool_features, neighbor_idx, keep_idx

    @property
    def k(self) -> int:
        return self.cluster.k

    @property
    def ratio(self) -> float:
        return self.decimate.ratio


class TransitionUp(nn.Module):
    def __init__(self, dim_coarse: int, dim_fine: int, act: nn.Module = nn.ReLU()):
        super().__init__()
        self.mlp_coarse = MLP(dim_coarse, dim_fine, dim_fine, act=act)
        self.mlp_fine = MLP(dim_fine, dim_fine, dim_fine, act=act)

    def forward(
        self, features_coarse: Tensor, features_fine: Tensor, neighbor_idx: List[Tensor], keep_idx: List[Tensor]
    ) -> Tensor:
        Lc, N, Dc = features_coarse.shape
        Lf, N, Df = features_fine.shape

        features_coarse = self.mlp_coarse(features_coarse)
        features_fine = self.mlp_fine(features_fine)
        D = Dc = Df
        assert tuple(features_coarse.shape) == (Lc, N, D)
        assert tuple(features_fine.shape) == (Lf, N, D)

        # TODO this wont handle a fine point that is in a cluster for multiple coarse points
        updated_features = features_coarse.view(1, Lc, N, D) + features_fine[neighbor_idx].view(-1, Lc, N, D)
        features_fine[neighbor_idx] = updated_features.view(-1, D)

        return features_fine
