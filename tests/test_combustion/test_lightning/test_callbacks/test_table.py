#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

import numpy as np
import pandas as pd
import pytest
import pytorch_lightning as pl
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from combustion.lightning.callbacks.table import DistributedDataFrame
from combustion.testing import cuda_or_skip


RANDOM_PORTS = list(np.random.randint(12000, 19000, 1000))


def set_random_main_port():
    pl.seed_everything(42)
    port = RANDOM_PORTS.pop()
    os.environ["MASTER_PORT"] = str(port)


def _test_collect_states(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    torch.cuda.set_device(f"cuda:{rank}")

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    df = DistributedDataFrame({"col1": rank, "col2": rank}, index=[rank])
    result = df.gather_all()
    assert len(result) == 2
    assert 0 in result.index
    assert 1 in result.index


class TestDistributedDataFrame:
    def test_init(self):
        rank = 0
        df = DistributedDataFrame({"col1": rank, "col2": rank}, index=[0])
        assert isinstance(df, pd.DataFrame)

    def test_no_process_group(self):
        rank = 0
        df = DistributedDataFrame({"col1": rank, "col2": rank}, index=[0])
        out = df.gather_all()
        assert (out == df).all().all()

    @cuda_or_skip
    @pytest.mark.skipif(condition=torch.cuda.device_count() < 2, reason="missing GPUs")
    def test_distributed_gather(self):
        """This test ensures state are properly collected across processes.
        This would be used to collect dataloader states as an example.
        """
        set_random_main_port()
        mp.spawn(_test_collect_states, args=(2,), nprocs=2)
