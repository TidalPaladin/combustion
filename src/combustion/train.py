#!/usr/bin/env python
# -*- coding: utf-8 -*-
import glob
import logging
import os
import warnings
from argparse import Namespace
from typing import Any, Optional, Tuple

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from ignite.contrib.metrics import ROC_AUC
from ignite.engine import Engine, Events, State
from ignite.exceptions import NotComputableError
from ignite.metrics import Accuracy, ConfusionMatrix, Fbeta, Loss, Precision, Recall, RunningAverage
from torch import Tensor
from torch.optim import Adam, Optimizer, RMSprop

from .data.loader import load_data
from .data.visual import visualize_image, visualize_mb
from .ignite.engine import SupervisedEvalFunc, SupervisedTrainFunc
from .ignite.handlers import (
    CheckpointLoader,
    LRScheduler,
    ModelCheckpoint,
    OutputVisualizer,
    ProgressBar,
    SummaryWriter,
    TrackedVisualizer,
    Tracker,
    Validate,
    ValidationLogger,
)
from .ignite.metrics import ScheduledLR
from .loss import get_criterion
from .model import get_model  # , regional_max


try:
    import apex
    from apex import amp
except ImportError:
    apex = None
    amp = None


def get_optim_from_args(args: Namespace, model: nn.Module) -> Optimizer:
    """get_optim_from_args
    Returns an optimizer based on supplied runtime flags.

    :param args: the runtime flags
    :type args: Namespace
    :param model: model with parameters to be optimized
    :type model: nn.Module
    :rtype: Optimizer
    :returns: Optimizer for `model` configured as per `args`
    """
    if args.optim == "adam":
        logging.debug("Got Adam optimizer")
        optim = Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), eps=args.epsilon)
    elif args.optim == "rmsprop":
        logging.debug("Got RMSprop optimizer")
        optim = RMSprop(model.parameters(), lr=args.lr, momentum=args.momentum, eps=args.epsilon)
    else:
        raise ValueError("unknown optimizer %s", args.optim)
    return optim


class TrainFunc(SupervisedTrainFunc):
    def step(self, engine, batch):
        # type: (Engine, Any) -> Tuple[Tensor, Tensor]
        inputs, labels = batch.frames.rename(None), batch.labels.rename(None)
        if len(labels.unique()) == 1:
            warnings.warn("Got only 1 label type for %s/%s" % (engine.state.iteration, engine.state.epoch))
        assert (labels <= 1).all()
        outputs = self.model(inputs)
        if outputs.ndim == 5:
            outputs = outputs.squeeze(-3)
        return outputs, labels


class EvalFunc(SupervisedEvalFunc):
    def step(self, engine, batch):
        # type: (Engine, Any) -> Tuple[Tensor, Tensor]
        inputs, labels = batch.frames.rename(None), batch.labels.rename(None)
        if len(labels.unique()) == 1:
            warnings.warn("Got only 1 label type for %s/%s" % (engine.state.iteration, engine.state.epoch))
        outputs = self.model(inputs)
        if outputs.ndim == 5:
            outputs = outputs.squeeze(-3)
        return outputs, labels


def visualize_fn(frame, label, true, title):
    return visualize_image(frame.numpy(), label.numpy(), true.numpy(), title=title, dpi=192)


def visualize_mb_fn(frame, label, true, title):
    return visualize_mb(frame.numpy(), label.numpy(), true.numpy(), title=title)


def preprocess_fn(state, threshold=None, suppression=None):
    batch_dim = -4
    channel_dim = -3
    with torch.no_grad():
        batch = state.batch
        pred_labels, true_labels = state.output
        pred_labels = torch.sigmoid(pred_labels)
        frames = state.batch.frames.rename(None)

        # slice out first example in batch dims suitable for visualization
        frames, label, true = frames[0], pred_labels[0], true_labels[0]
        label = label.squeeze(channel_dim)
        frame = frames[:, frames.shape[-3] // 2, ...]
        true = true.squeeze(channel_dim)

        # upsample input frame if in super resolution mode
        if frame.shape != label.shape:
            frame = F.interpolate(frame.unsqueeze(batch_dim), size=label.shape, mode="nearest")
            frame = frame.squeeze(batch_dim)
        frame = frame.squeeze(channel_dim)

        if suppression is not None:
            # label = regional_max(label.unsqueeze(channel_dim), suppression).squeeze(channel_dim)
            pass
        if threshold is not None:
            label = (label >= threshold).to(label)
    return frame, label, true


# function for thresholding
def thresh_fn(threshold):
    def wrapped(output):
        with torch.no_grad():
            ypred, ytrue = output
            ypred = torch.sigmoid(ypred)
            # ypred = regional_max(ypred, 12)
            pass
            ypred = (ypred >= threshold).byte()
            return ypred.flatten(), ytrue.flatten()

    return wrapped


def train(args: Namespace) -> State:
    ngpus_per_node = args.gpus_per_node
    cudnn.benchmark = True

    # get model
    model = get_model(args)
    if args.gpu is not None:
        model.cuda(args.gpu)
    else:
        model.cuda()

    optimizer = get_optim_from_args(args, model)

    if args.opt_level is not None:
        logging.info("Using amp at opt level %s", args.opt_level)
        # amp.register_float_function(torch, 'sigmoid')
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level)

    lr_scheduler = LRScheduler.from_args(args, optimizer)

    if args.distributed:
        if args.gpu is not None:
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)

        if args.opt_level is None:
            if args.gpu is not None:
                logging.info("Applying torch DDParallel wrapper")
                model = torch.nn.parallel.DistributedDataParallel(
                    model, device_ids=[args.gpu], find_unused_parameters=True
                )
            else:
                model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
        else:
            logging.info("Applying apex DDParallel wrapper")
            model = apex.parallel.DistributedDataParallel(model)
    elif args.rank != -1:
        logging.info("Applying torch DataParallel wrapper")
        model = torch.nn.DataParallel(model)
    else:
        logging.info("Not applying distributed wrapper")

    # get datasets
    train_ds, val_ds = load_data(args, "train")
    logging.info("Batches in training set: %d", len(train_ds))
    logging.info("Batches in validation set: %s", len(val_ds) if val_ds is not None else "None")
    args.steps_per_epoch = len(train_ds)

    for batch in train_ds:
        inputs, labels = batch.frames.rename(None), batch.labels.rename(None)
        assert len(labels.nonzero()) > 0

    # get criterion
    train_criterion, eval_criterion = get_criterion(args)
    logging.info("Train criterion: %s", train_criterion.__class__.__name__)
    logging.info("Eval criterion: %s", eval_criterion.__class__.__name__)

    # setup ignite engines
    process_fn = TrainFunc.from_args(
        args, model, optimizer, train_criterion, device=args.gpu, lr_schedule=lr_scheduler
    )
    eval_fn = EvalFunc(model, device=args.gpu)
    trainer = Engine(process_fn)
    train_eval = Engine(eval_fn)
    val_eval = Engine(eval_fn)

    to_save = {"model": model, "optimizer": optimizer, "trainer": trainer}
    if args.opt_level is not None:
        to_save["amp"] = amp

    # attach loss / accuracy metrics. use BCELoss when model.eval() is set
    RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")
    Loss(train_criterion).attach(train_eval, "loss")
    Loss(train_criterion).attach(val_eval, "loss")

    def fn2(output):
        with torch.no_grad():
            ypred, ytrue = output
            ypred = torch.sigmoid(ypred)
            if len(ytrue.unique()) <= 1:
                warnings.warn("Encountered y_pred of one class for ROC AUC")
                ypred = torch.Tensor([0.55, 0.45])
                ytrue = torch.Tensor([1, 0])
            return ypred.flatten(), ytrue.flatten()

    ROC_AUC(output_transform=fn2).attach(val_eval, f"roc_auc")

    val_fmt = "Validation Results - Epoch: {epoch} Loss: {loss:.5E}, ROC_AUC: {roc_auc:.5f}"

    # attach metrics for multiple thresholds
    if args.threshold:
        logging.info("Attaching thresholded metrics for %s", args.threshold)

        # Build dict of metrics and validation output string
        thresh_metrics = {}
        for i, threshold in enumerate(args.threshold):
            fn = thresh_fn(threshold)

            thresh_metrics.update(
                {
                    f"acc_{i}": Accuracy(output_transform=fn),
                    f"precision_{i}": Precision(output_transform=fn),
                    f"recall_{i}": Recall(output_transform=fn),
                    f"fmeasure_{i}": Fbeta(beta=1, output_transform=fn),
                }
            )

            # modify format for validation outputs
            val_fmt += f"\n=== Thresh={threshold} ==="
            val_fmt += "\n\tAccuracy: {acc_" + str(i) + ":.5f}"
            val_fmt += "\n\tPrecision: {precision_" + str(i) + ":.5E}"
            val_fmt += "\n\tRecall: {recall_" + str(i) + ":.5E}"
            val_fmt += "\n\tFMeasure: {fmeasure_" + str(i) + ":.5E}"
        val_fmt += "\n===="

        # attach metrics from dict
        for k, v in thresh_metrics.items():
            v.attach(val_eval, k)

    # attach handlers
    if lr_scheduler is not None:
        lr_scheduler.attach(trainer)
        to_save["lr_scheduler"] = lr_scheduler
        ScheduledLR(lr_scheduler).attach(trainer, "lr")

    restore = CheckpointLoader.from_args(args, **to_save)

    if args.multiprocessing_distributed and args.rank % args.gpus_per_node != 0:
        pass
    else:
        summary = SummaryWriter.from_args(args, model, transform=lambda x: x.frames.rename(None))
        summary.attach(trainer)

        checkpoint = ModelCheckpoint.from_args(args)
        checkpoint.attach(trainer, args, **to_save)

        OutputVisualizer.from_args(args, lambda x: preprocess_fn(x, None), visualize_fn, dpi=142).attach(
            val_eval, event=Events.ITERATION_COMPLETED(every=5)
        )
        OutputVisualizer.from_args(
            args,
            lambda x: preprocess_fn(x, 0.5, 12),
            visualize_fn,
            fmt="output_thresh_{epoch}_{iteration}.png",
            dpi=142,
        ).attach(val_eval, event=Events.ITERATION_COMPLETED(every=5))
        OutputVisualizer.from_args(args, lambda x: preprocess_fn(x, None), visualize_mb_fn,).attach(
            val_eval, event=Events.ITERATION_COMPLETED(every=5)
        )
        OutputVisualizer.from_args(
            args, lambda x: preprocess_fn(x, 0.5, 12), visualize_mb_fn, fmt="output_thresh_{epoch}_{iteration}.png",
        ).attach(val_eval, event=Events.ITERATION_COMPLETED(every=5))

        Tracker().attach(trainer, Events.ITERATION_COMPLETED(every=20), ["loss", "lr"])
        vis2 = TrackedVisualizer(args.result_path, "loss_lr.png", ["loss", "lr"], title="Loss versus Learning Rate")
        vis2.attach(trainer, event=Events.ITERATION_COMPLETED(every=100))

    progbar = ProgressBar.from_args(args, metrics=["loss"])
    if progbar is not None:
        progbar.attach(trainer)

    val_handler = ValidationLogger(val_fmt, progbar)
    validation_cb = Validate(val_eval, val_ds, model)
    val_handler.attach(val_eval)

    if val_ds:
        logging.info("Added validation callback")
        validation_cb.attach(trainer)
    else:
        logging.info("No validation data given, skipping...")

    if restore is None:
        logging.info("Starting fresh training run")
        state = trainer.run(train_ds, max_epochs=args.epochs, seed=args.seed)
    else:
        restore(trainer)
        state = trainer.run(train_ds)

    logging.info("Finished training run")
    return state
