#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Any, List, Optional, TypeVar, cast

import pandas as pd
import torch.distributed as dist


T = TypeVar("T")


def all_gather_object(obj: T, group: Optional[Any] = None) -> List[T]:
    if group is None:
        group = dist.group.WORLD

    world_size = dist.get_world_size(group)
    gathered_result = [None for _ in range(world_size)]

    # sync and broadcast all
    dist.barrier(group=group)
    dist.all_gather_object(gathered_result, obj, group)

    return cast(List[T], gathered_result)


class DistributedDataFrame(pd.DataFrame):
    def gather_all(self, group: Optional[Any] = None) -> pd.DataFrame:
        r"""Gather this distributed dataframe across processes"""
        gathered = all_gather_object(self, group)
        return pd.concat(gathered)
