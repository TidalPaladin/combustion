#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import os
import random
import warnings
from argparse import Namespace

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

try:
    from sr_mb.data.visual import check_data
    from sr_mb.evaluate import test
    from sr_mb.flags import parse_args
    from sr_mb.train import train
except ImportError:
    from combustion.data.visual import check_data
    from combustion.evaluate import test
    from combustion.args import parse_args
    from combustion.train import train


def init_logger(args: Namespace):
    """init_logger
    initializes the python logging module based on flags 
    provided at runtime

    :param args: the argparse namespace
    """
    log_level = "DEBUG" if args.verbose else "INFO"
    log_format = logging.Formatter(args.log_format)
    logger = logging.getLogger()
    logger.setLevel(log_level)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    if args.log_file and args.log_file != "" and not args.dry:
        file_handler = logging.FileHandler(args.log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    return logger


def check_empty_dirs(args: Namespace):
    if os.path.exists(args.result_path) and os.listdir(args.result_path):
        raise RuntimeError("result_path must be empty when --overwrite not used")
    if os.path.exists(args.model_path) and os.listdir(args.model_path):
        raise RuntimeError("model_path must be empty when --overwrite not used")


def setup(args):
    args = parse_args(parser=None, args=args)
    init_logger(args)

    logging.info("Parsed args")
    for k, v in vars(args).items():
        logging.info("%s = %s", k, str(v))

    if not args.overwrite:
        try:
            check_empty_dirs(args)
        except RuntimeError as e:
            logging.exception("Directory not empty and --overwrite not given")

    if args.seed is not None:
        logging.info("Setting seed %d", args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )

    if args.visible_gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(x) for x in args.visible_gpus])
        logging.info("Set CUDA_VISIBLE_DEVICES: %s", os.environ["CUDA_VISIBLE_DEVICES"])

    if args.gpu is not None:
        warnings.warn("You have chosen a specific GPU. This will completely " "disable data parallelism.")

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()
    setattr(args, "gpus_per_node", ngpus_per_node)

    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size

    logging.info("Finished early setup")
    return args


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.gpu is not None:
        logging.info("Use GPU: %d" % args.gpu)
        torch.cuda.set_device(args.gpu)

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
        )
        logging.info("Initialized process group")

    if args.mode == "test":
        logging.debug("Entering testing loop")
        test(args)
    elif args.mode == "train":
        logging.debug("Entering training loop")
        train(args)
    elif args.mode == "check_data":
        logging.debug("Running check_data")
        check_data(args)
    else:
        raise NotImplementedError(args.mode)
    logging.info("Exiting...")
