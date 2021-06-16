#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch import Tensor
import torch
import torch.nn as nn
from typing import Tuple

import torch_scatter as ts
import torch_cluster as tc

def get_batch_idx(x: Tensor) -> Tensor:
    N, L = x.shape[:2]
    batch_idx = torch.arange(N).to(x.device).view(N, 1).expand(N, L)
    return batch_idx.long()

class NearestNeighborCluster(nn.Module):

    def __init__(self, k: int):
        super().__init__()
        self.k = k
    
    def forward(self, coords: Tensor, features: Tensor) -> Tensor:
        N, L, D = features.shape
        batch_idx = get_batch_idx(coords)

        batch_idx = batch_idx.reshape(N * L)
        coords = coords.view(N * L, -1)
        edge_index = tc.knn_graph(coords, self.k, batch=batch_idx, loop=True)
        dests, sources = (x.view(-1) for x in edge_index.chunk(2, dim=0))
        features = features.view(-1, D)
        out = features[dests].view(self.k, N*L, D).swapdims(0, 1)
        return out

class FarthestPointsReduce(nn.Module):

    def __init__(self, ratio: float):
        super().__init__()
        self.ratio = ratio
    
    def forward(self, coords: Tensor, features: Tensor) -> Tuple[Tensor, Tensor]:
        N, L, C = coords.shape
        N2, K, D = features.shape

        batch_idx = get_batch_idx(coords).reshape(N*L)
        coords = coords.view(N * L, -1)
        index = tc.fps(coords, batch_idx, self.ratio)

        coords = coords[index].view(N, -1, C)


        assert False
