#!/usr/bin/env python
import logging
import warnings

import torch.multiprocessing as mp

from .setup import main_worker, setup


def main(args=None):
    args = setup(args)

    if args.multiprocessing_distributed and args.rank == -1:
        raise ValueError("--rank is required with --multiprocessing_distributed")

    if args.gpu is not None and args.rank != -1:
        warnings.warn("both --gpu and --rank are given, be careful")

    if args.multiprocessing_distributed:
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        logging.info("Spawning processes")
        mp.spawn(main_worker, nprocs=args.gpus_per_node, args=(args.gpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, args.gpus_per_node, args)

    if args.multiprocessing_distributed:
        logging.info("Exiting distributed main process")


if __name__ == "__main__":
    main()
