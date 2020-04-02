#!python
"""Provides command line flags to customize the training pipleine"""

import argparse
import os
import sys
import time
from argparse import ArgumentParser

from .validators import NumericValidator
from .vision import add_vision_args


# Read env vars for some defaults
def setup_parser() -> argparse.Namespace:
    """parse_args
    Creates an argparse parser and invokes parser.parse_args().

    :param args: input cli flags to parse. default to sys.argv[1:]
    :rtype: argparse.Namespace
    """

    parser = argparse.ArgumentParser(
        description=(
            "template for dockerized deep learning applications. "
            "some common flags are provided. "
            "see https://github.com/TidalPaladin/dockerized-deep-learning for "
            "more information."
        )
    )
    add_preprocessing_args(parser)
    add_ignite_args(parser)
    add_file_args(parser)
    add_logging_args(parser)
    add_checkpoint_args(parser)
    add_classification_args(parser)
    add_regularization_args(parser)
    add_model_args(parser)
    add_runtime_args(parser)
    add_distributed_args(parser)
    add_validation_args(parser)
    add_optimizer_args(parser)
    add_criterion_args(parser)
    add_training_args(parser)
    add_lr_schedule_args(parser)
    add_vision_args(parser)
    return parser


# Read env vars for some defaults
def parse_args(parser=None, args=None) -> argparse.Namespace:
    """parse_args
    Creates an argparse parser and invokes parser.parse_args().

    :param args: input cli flags to parse. default to sys.argv[1:]
    :rtype: argparse.Namespace
    """
    if parser is None:
        parser = setup_parser()
    args = parser.parse_args(args)
    return _postprocess(args)


def add_preprocessing_args(parser: ArgumentParser) -> ArgumentParser:
    """Adds general preprocessing args to a parser"""
    group = parser.add_argument_group("preprocessing")
    norm_strat = group.add_mutually_exclusive_group()
    norm_strat.add_argument(
        "--normalize", default=False, action="store_true", help="normalize inputs to the range [0, 1]",
    )
    norm_strat.add_argument(
        "--standardize", default=False, action="store_true", help="standardize inputs to zero mean unit variance",
    )
    return group


def add_ignite_args(parser: ArgumentParser) -> ArgumentParser:
    """Adds args for the high level Ignite library to a parser"""
    group = parser.add_argument_group("ignite config")
    group.add_argument("--progbar", default=False, action="store_true", help="output a progress bar")
    group.add_argument("--progbar_format", default=None, help="format string for the progress bar")
    group.add_argument("--gpuinfo", default=False, action="store_true", help="track GPU info (memory, etc.)")
    group.add_argument("--gpu_format", default="gpu:0 mem(%)", help="format for GPU info output. default %(default)s")
    return group


def add_tensorboard_args(parser: ArgumentParser) -> ArgumentParser:
    """Adds args used for Tensorboard control"""
    # default tb_logs path based on env vars
    ARTIFACT_DIR = os.environ.get("ARTIFACT_DIR", "./artifacts")
    default_tb_path = os.path.join(ARTIFACT_DIR, "tb_logs")

    group = parser.add_argument_group("tensorboard config")
    group.add_argument(
        "--tensorboard_dir",
        type=str,
        metavar="PATH",
        default=default_tb_path,
        help="path to write tensorboard logs. default %(default)s",
    )
    group.add_argument(
        "--write_graph",
        default=False,
        action="store_true",
        help="write graph to tensorboard logs. default %(default)s",
    )
    group.add_argument(
        "--write_images",
        default=False,
        action="store_true",
        help="write images to tensorboard logs. default %(default)s",
    )
    return group


def add_file_args(parser: ArgumentParser) -> ArgumentParser:
    """Adds args for file/artifact IO"""
    # select default directories from env variables
    DATA_DIR = os.environ.get("DATA_DIR", "./data")
    ARTIFACT_DIR = os.environ.get("ARTIFACT_DIR", "./artifacts")

    # File path selection
    group = parser.add_argument_group(
        "file paths",
        description=(
            "paths to input data and output runtime artifacts. "
            "default values are assigned as subdirectories of the "
            "DATA_DIR and ARTIFACT_DIR environment vars"
        ),
    )
    group.add_argument(
        "--data_path", default=DATA_DIR, metavar="PATH", help="filepath of dataset. default %(default)s",
    )
    group.add_argument(
        "--output_path", default=DATA_DIR, metavar="PATH", help="path to store runtime artifacts. default %(default)s",
    )
    group.add_argument(
        "--temp_path", default=DATA_DIR, metavar="PATH", help="path to store temporary files. default %(default)s",
    )
    group.add_argument(
        "--overwrite", default=False, action="store_true", help="if set, overwrite files on conflict",
    )
    return group


def add_logging_args(parser: ArgumentParser) -> ArgumentParser:
    """Adds args to control logging"""
    group = parser.add_argument_group("logging")
    group.add_argument(
        "--log_file",
        type=str,
        metavar="FILE",
        default=None,
        help="log file name. default MODEL_MODE_TIMESTAMP.log.txt",
    )
    group.add_argument(
        "--log_format",
        type=str,
        metavar="STRING",
        default="[%(asctime)s %(levelname).1s] - %(message)s",
        help=("log record format string. see python logging documentation for available options"),
    )
    group.add_argument("-v", "--verbose", action="count", default=0, help="sets logging verbosity")
    return group


def add_checkpoint_args(parser: ArgumentParser) -> ArgumentParser:
    group = parser.add_argument_group("checkpointing")

    load_group = group.add_mutually_exclusive_group()
    load_group.add_argument(
        "--resume", default=False, action="store_true", help="resume training from most recent checkpoint"
    )
    load_group.add_argument(
        "--load_checkpoint", default=None, type=str, metavar="FILE", help="load checkpoint from checkpoint file"
    )

    checkpoint_freq = group.add_mutually_exclusive_group()
    checkpoint_freq.add_argument(
        "--checkpoint_steps",
        default=None,
        type=int,
        low=1,
        metavar="N",
        inclusive=(True, True),
        action=NumericValidator,
        help="create model checkpoint every N steps.",
    )
    checkpoint_freq.add_argument(
        "--checkpoint_epochs",
        default=None,
        type=int,
        low=1,
        metavar="N",
        inclusive=(True, True),
        action=NumericValidator,
        help="create model checkpoint every N epochs.",
    )

    group.add_argument(
        "--checkpoint_prefix",
        default=None,
        type=str,
        metavar="STRING",
        help="prefix for checkpoint files. by default, use value of --model",
    )
    group.add_argument(
        "--n_saved",
        default=None,
        type=int,
        low=1,
        metavar="N",
        action=NumericValidator,
        help="number of checkpoints to keep. by default, keep all checkpoints.",
    )
    return group


def add_classification_args(parser: ArgumentParser) -> ArgumentParser:
    group = parser.add_argument_group("classification")
    group.add_argument(
        "--num_classes",
        default=64,
        type=int,
        low=1,
        action=NumericValidator,
        metavar="N",
        help="number of classes for classification. default %(default)s",
    )
    group.add_argument(
        "--threshold",
        nargs="+",
        type=float,
        low=0,
        high=1.0,
        inclusive=(False, False),
        action=NumericValidator,
        help=(
            "decision threshold rounding probabilites to a class."
            " specify more than one threshold if desired."
            " default %(default)s"
        ),
    )
    return group


def add_regularization_args(parser: ArgumentParser) -> ArgumentParser:
    group = parser.add_argument_group("regularization")
    # TODO if patch approach works well, add patching to l1/l2
    group.add_argument(
        "--l1",
        default=0,
        type=float,
        low=0,
        high=1.0,
        inclusive=(True, False),
        action=NumericValidator,
        help="l1 regularization value. default %(default)s",
    )
    group.add_argument(
        "--l2",
        default=0,
        type=float,
        low=0,
        high=1.0,
        inclusive=(True, False),
        action=NumericValidator,
        help="l2 regularization value. default %(default)s",
    )

    group.add_argument(
        "--dropout",
        default=0,
        type=float,
        low=0,
        high=1.0,
        inclusive=(True, False),
        action=NumericValidator,
        help="dropout ratio. default %(default)s",
    )
    return group


def add_model_args(parser: ArgumentParser) -> ArgumentParser:
    group = parser.add_argument_group("model properties")
    group.add_argument(
        "model",
        default="model1",
        type=str,
        choices=["model1", "model2"],
        help="model architecture to use. default %(default)s",
    )
    group.add_argument(
        "--width",
        default=32,
        type=int,
        low=1,
        action=NumericValidator,
        help="model width after tail. default %(default)s",
    )

    fact_group = group.add_mutually_exclusive_group()
    fact_group.add_argument(
        "--factorized",
        default=False,
        action="store_true",
        help="set to use a factorized model. defaults to model's preferred choice",
    )
    fact_group.add_argument(
        "--unfactorized",
        default=False,
        action="store_true",
        help="set to use an unfactorized model. defaults to model's preferred choice",
    )

    group.add_argument(
        "--checkpoint",
        nargs="+",
        default=None,
        help=(
            "class or layer names to checkpoint."
            " intermediate activations for checkpointed layers are recomputed, rather than stored.",
            " this trades lower memory usage for increased compute.",
        ),
    )
    group.add_argument(
        "--checkpoint_targets", nargs="+", help="nn.Module subclasses or layer names to patch with checkpointing",
    )
    return group


def add_runtime_args(parser: ArgumentParser) -> ArgumentParser:
    # runtime properties
    group = parser.add_argument_group("runtime", description="args controlling miscellaneous runtime properties")
    group.add_argument("--mode", default="train", type=str, choices=["train", "test", "profile", "check_data"])
    group.add_argument(
        "--opt_level",
        default=None,
        choices=["O0", "O1", "O2", "O3"],
        help="optimization level for Nvidia's AMP. requires default %(default)s",
    )
    group.add_argument("--batch_size", default=32, type=int, low=1, metavar="N", action=NumericValidator)
    group.add_argument(
        "--visible_gpus",
        default=None,
        type=int,
        nargs="+",
        metavar="N",
        help=(
            "gpus visible for execution. will be used to update "
            "CUDA_VISIBLE_DEVICES. by default, all gpus are visible."
        ),
    )
    return group


def add_distributed_args(parser: ArgumentParser) -> ArgumentParser:
    group = parser.add_argument_group("distributed", description="args controlling multi-gpu and multi-node execution")
    group.add_argument(
        "--world_size",
        type=int,
        default=-1,
        low=1,
        inclusive=(True, False),
        action=NumericValidator,
        metavar="N",
        help="number of machines to be used in training. default %(default)s",
    )
    group.add_argument("--rank", default=-1, type=int, help="node rank for distributed training")
    group.add_argument("--dist-url", default=None, type=str, help="url used to set up distributed training")
    group.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
    group.add_argument("--gpu", default=None, type=int, help="GPU id to use.")
    group.add_argument(
        "--multiprocessing-distributed",
        action="store_true",
        help="Use multi-processing distributed training to launch "
        "N processes per node, which has N GPUs. This is the "
        "fastest way to use PyTorch for either single node or "
        "multi node data parallel training",
    )
    group.add_argument(
        "-j", "--workers", default=4, type=int, metavar="N", help="number of data loading workers (default: 4)"
    )

    group.add_argument("--seed", default=None, type=int, metavar="N", help="integer seed for random generators")
    group.add_argument("--dry", default=False, action="store_true", help="run without generating artifacts")
    group.add_argument("--summary", default=False, action="store_true", help="print model.summary() and exit")
    group.add_argument("--tune", default=False, action="store_true", help="reserved for future use")
    group.add_argument("--image_dim", type=int, low=1, action=NumericValidator, nargs=2, metavar="N", default=[64, 64])
    return group


def add_validation_args(parser: ArgumentParser) -> ArgumentParser:
    group = parser.add_argument_group("validation")

    source = group.add_mutually_exclusive_group()
    source.add_argument(
        "--val_split",
        default=None,
        type=float,
        low=0,
        high=1,
        inclusive=False,
        action=NumericValidator,
        help="fraction of training data to reserve for validation",
    )
    source.add_argument(
        "--val_path", default=None, help="filepath to validation dataset. default %(default)s",
    )

    group.add_argument(
        "--val_steps",
        default=None,
        type=int,
        low=1,
        action=NumericValidator,
        help="number of validation steps. by default, use all validation data if given",
    )
    group.add_argument(
        "--val_output_path",
        default="epoch_{epoch}/output_{epoch}_{iteration}.png",
        help="format for validation example outputs. default %(default)s",
    )
    group.add_argument(
        "--metrics",
        nargs="+",
        choices=["accuracy", "mse", "conf_mat", "precision", "recall", "roc_auc",],
        help="metrics to record during validation / testing",
    )
    return group


def add_optimizer_args(parser: ArgumentParser) -> ArgumentParser:
    group = parser.add_argument_group("optimizer")
    group.add_argument("--optim", default="adam", type=str, choices=["adam", "rmsprop"], help="optimizer selection")
    group.add_argument(
        "--grad_steps",
        default=1,
        type=int,
        low=1,
        inclusive=(True, True),
        metavar="VAL",
        action=NumericValidator,
        help="compute gradients every VAL steps. default %(default)s",
    )
    group.add_argument(
        "--grad_clip",
        default=None,
        type=float,
        low=0,
        inclusive=(False, True),
        metavar="VAL",
        action=NumericValidator,
        help="gradient clipping by norm on model weights. default %(default)s",
    )
    group.add_argument(
        "--out_grad_clip",
        default=None,
        type=float,
        low=0,
        inclusive=(False, True),
        metavar="VAL",
        action=NumericValidator,
        help="gradient clipping by norm on outputs. default %(default)s",
    )
    group.add_argument(
        "--lr",
        default=0.001,
        type=float,
        low=0,
        high=1.0,
        inclusive=(False, False),
        metavar="VAL",
        action=NumericValidator,
        help="learning rate. default %(default)s",
    )
    group.add_argument(
        "--beta1",
        default=0.9,
        type=float,
        low=0,
        high=1.0,
        inclusive=(False, False),
        metavar="VAL",
        action=NumericValidator,
        help="adam beta1. default %(default)s",
    )
    group.add_argument(
        "--beta2",
        default=0.999,
        type=float,
        low=0,
        high=1.0,
        inclusive=(False, False),
        metavar="VAL",
        action=NumericValidator,
        help="adam beta2. default %(default)s",
    )
    group.add_argument(
        "--epsilon",
        default=1e-6,
        type=float,
        low=0,
        high=1.0,
        inclusive=(False, False),
        metavar="VAL",
        action=NumericValidator,
        help="optimizer epsilon value. default %(default)s",
    )
    group.add_argument(
        "--momentum",
        default=0.0,
        type=float,
        low=0,
        high=1.0,
        metavar="VAL",
        action=NumericValidator,
        help="optimizer momentum value. default %(default)s",
    )
    group.add_argument(
        "--rho",
        default=0.9,
        type=float,
        low=0,
        high=1.0,
        metavar="VAL",
        action=NumericValidator,
        help="optimizer rho value. default %(default)s",
    )
    return group


def add_criterion_args(parser: ArgumentParser) -> ArgumentParser:
    group = parser.add_argument_group("criterion", "loss function settings")
    group.add_argument(
        "criterion",
        type=str,
        choices=["mse", "bce", "wmse", "wbce", "focal"],
        help="loss function to use. default %(default)s",
    )
    group.add_argument(
        "--smoothing_sigma",
        default=None,
        type=float,
        low=0,
        metavar="VAL",
        action=NumericValidator,
        inclusive=(False, True),
        help="smoothed MSE loss sigma value. default %(default)s",
    )
    group.add_argument(
        "--smoothing_kernel",
        default=None,
        type=int,
        low=1,
        metavar="VAL",
        action=NumericValidator,
        help="smoothed MSE loss gaussian kernel value. default %(default)s",
    )
    group.add_argument(
        "--sparsity",
        default=0,
        type=float,
        low=0,
        metavar="VAL",
        action=NumericValidator,
        help="smoothed MSE loss sparsity reward coefficient. default %(default)s",
    )
    group.add_argument(
        "--max_weight",
        default=None,
        type=float,
        low=0.0,
        metavar="VAL",
        action=NumericValidator,
        inclusive=(False, True),
        help="maximum weight for any element when using weighted loss. default %(default)s",
    )
    group.add_argument(
        "--reduction",
        default="mean",
        choices=["mean", "sum"],
        help="loss function reduction method. default %(default)s",
    )
    group.add_argument(
        "--focal_alpha",
        default=None,
        type=float,
        low=0.0,
        metavar="VAL",
        action=NumericValidator,
        inclusive=(False, True),
        help="focal loss alpha value. default %(default)s",
    )
    group.add_argument(
        "--focal_gamma",
        default=None,
        type=float,
        low=0.0,
        metavar="VAL",
        action=NumericValidator,
        inclusive=(False, True),
        help="focal loss gamma value. default %(default)s",
    )
    group.add_argument(
        "--focal_smoothing",
        default=None,
        type=float,
        low=0.0,
        high=1.0,
        metavar="VAL",
        action=NumericValidator,
        inclusive=(True, True),
        help="focal loss label smoothing value. default %(default)s",
    )

    # initialization properties
    init_group = parser.add_argument_group("initializer", "override weight initialization strategy")
    init_alg = init_group.add_mutually_exclusive_group()
    init_alg.add_argument(
        "--glorot_normal", default=False, action="store_true", help="use glorot/xavier normal initialization"
    )
    init_alg.add_argument(
        "--glorot_uniform", default=False, action="store_true", help="use glorot/xavier uniform initialization"
    )
    return group

    # training properties


def add_training_args(parser: ArgumentParser) -> ArgumentParser:
    group = parser.add_argument_group("training")
    train_len = group.add_mutually_exclusive_group()
    train_len.add_argument(
        "--steps",
        default=None,
        type=int,
        low=1,
        metavar="N",
        action=NumericValidator,
        help="number of steps to train for",
    )
    train_len.add_argument(
        "--epochs",
        default=None,
        type=int,
        low=1,
        metavar="N",
        action=NumericValidator,
        help="number of epochs to train for",
    )
    group.add_argument(
        "--steps_per_epoch",
        default=None,
        type=int,
        low=1,
        metavar="N",
        action=NumericValidator,
        help="training steps per epoch",
    )

    group.add_argument(
        "--early_stopping", default=False, action="store_true",
    )
    return group


def add_lr_schedule_args(parser: ArgumentParser) -> ArgumentParser:
    group = parser.add_argument_group("learning rate schedule")
    group.add_argument(
        "--lr_decay",
        default=None,
        nargs="+",
        choices=["one_cycle", "exp", "cos_ann"],
        help="learning rate decay strategy",
    )
    group.add_argument(
        "--warmup_percent",
        default=0.3,
        type=float,
        low=0,
        high=1.0,
        inclusive=(False, False),
        metavar="VAL",
        action=NumericValidator,
        help="percentage of training to spend on lr warmup. default %(default)s",
    )
    group.add_argument(
        "--div_factor",
        default=25,
        type=float,
        low=0,
        inclusive=(False, True),
        metavar="VAL",
        action=NumericValidator,
        help="in one_cycle scheduling, determine initial lr as --lr / --div_factor. default %(default)s",
    )
    group.add_argument(
        "--final_div_factor",
        default=1e4,
        type=float,
        low=0,
        inclusive=(False, True),
        metavar="VAL",
        action=NumericValidator,
        help="in one_cycle scheduling, determine final lr as --lr / --div_factor. default %(default)s",
    )
    group.add_argument(
        "--lr_base_momentum",
        default=0.85,
        type=float,
        low=0,
        inclusive=(False, True),
        metavar="VAL",
        action=NumericValidator,
        help="base momentum for cycled lr schedules. default %(default)s",
    )
    group.add_argument(
        "--lr_max_momentum",
        default=0.95,
        type=float,
        low=0,
        inclusive=(False, True),
        metavar="VAL",
        action=NumericValidator,
        help="max momentum for cycled lr schedules. default %(default)s",
    )
    group.add_argument(
        "--lr_decay_coeff",
        default=None,
        type=float,
        low=0,
        metavar="VAL",
        inclusive=(False, True),
        action=NumericValidator,
        help="coefficient of exponential learning rate decay",
    )
    group.add_argument(
        "--lr_decay_steps",
        default=None,
        type=int,
        low=1,
        metavar="VAL",
        inclusive=(True, True),
        action=NumericValidator,
        help="steps in lr decay cycle",
    )
    group.add_argument(
        "--lr_cycle_len",
        default=None,
        type=int,
        low=1,
        metavar="VAL",
        inclusive=(True, True),
        action=NumericValidator,
        help="lr decay cycle length",
    )
    group.add_argument(
        "--lr_anneal",
        default="cos",
        choices=["cos", "linear"],
        help="lr schedule annealing strategy. default %(default)s",
    )
    group.add_argument(
        "--lr_decay_freq",
        default=10,
        type=int,
        low=1,
        metavar="N",
        action=NumericValidator,
        help="epoch frequency of decay step function",
    )
    group.add_argument(
        "--lr_cycle_mult", default=None, type=float, metavar="N", help="cycle length multiplier for lr decay",
    )
    group.add_argument(
        "--lr_start_mult",
        default=None,
        type=float,
        low=0,
        metavar="N",
        inclusive=(False, True),
        action=NumericValidator,
        help="TODO",
    )
    group.add_argument(
        "--lr_end_mult",
        default=None,
        type=float,
        low=0,
        metavar="N",
        action=NumericValidator,
        inclusive=(False, True),
        help="TODO",
    )
    group.add_argument(
        "--lr_save_hist", default=False, action="store_true", help="TODO",
    )
    return group


def _postprocess(args: argparse.Namespace) -> argparse.Namespace:
    # identify run w/ model/mode/timestamp
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    run_identifier = "%s_%s_%s" % (args.model, args.mode, current_time)

    if hasattr(args, "tensorboard_dir"):
        args.tensorboard_dir = os.path.join(args.tensorboard_dir, run_identifier)

    # TODO add hasattr checks to handle add_XXX not called
    if args.visible_gpus is not None:
        os.environ["CUDA_VISIBLE_DEIVCES"] = str(args.visible_gpus)

    if args.checkpoint_prefix is None:
        args.checkpoint_prefix = args.model

    if args.steps and args.steps_per_epoch:
        args.epochs = args.steps // args.steps_per_epoch

    if not args.factorized and not args.unfactorized:
        args.factorized = None

    if args.opt_level is not None:
        try:
            import apex
        except ImportError:
            raise ImportError(
                ("--opt_level requires apex, see https://github.com/NVIDIA/apex" " for installation instructions")
            )

    result_path = os.path.join(args.data_path, "results", run_identifier)
    setattr(args, "result_path", result_path)

    model_path = os.path.join(args.data_path, "models", run_identifier)
    setattr(args, "model_path", model_path)

    if args.log_file is None:
        args.log_file = "%s.log.txt" % run_identifier
    args.log_file = os.path.join(args.data_path, args.log_file)
    return args
