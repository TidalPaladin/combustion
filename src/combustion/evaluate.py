#!/usr/bin/env python
import glob
import logging
import os
from argparse import Namespace
from typing import Any, Optional, Tuple

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from ignite.contrib.metrics import ROC_AUC
from ignite.engine import Engine, Events, State
from ignite.metrics import Loss, Precision, Recall, RunningAverage
from torch import Tensor
from torch.optim import Adam, Optimizer, RMSprop

from .data.loader import load_data
from .data.visual import visualize_image
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
from .model import get_model


try:
    import apex
    from apex import amp
except ImportError:
    apex = None
    amp = None


class TrainFunc(SupervisedTrainFunc):
    def step(self, engine, batch):
        # type: (Engine, Any) -> Tuple[Tensor, Tensor]
        inputs, labels = batch.frames.rename(None), batch.labels.rename(None)
        outputs = self.model(inputs)
        if outputs.ndim == 5:
            outputs = outputs.squeeze(-3)
        return outputs, labels


class EvalFunc(SupervisedEvalFunc):
    def step(self, engine, batch):
        # type: (Engine, Any) -> Tuple[Tensor, Tensor]
        inputs, labels = batch.frames.rename(None), batch.labels.rename(None)
        outputs = self.model(inputs)
        if outputs.ndim == 5:
            outputs = outputs.squeeze(-3)
        return outputs, labels


def visualize_fn(frame, label, true, title):
    return visualize_image(frame.numpy(), label.numpy(), true.numpy(), title=title, dpi=192)


def preprocess_fn(state, threshold=None):
    batch_dim = -4
    channel_dim = -3
    with torch.no_grad():
        batch = state.batch
        pred_labels, true_labels = state.output
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

        if threshold is not None:
            label = (label >= threshold).to(label)
    return frame, label, true


def test(args: Namespace) -> State:
    ngpus_per_node = args.gpus_per_node
    cudnn.benchmark = True

    # get model
    model = get_model(args)
    if args.gpu is not None:
        model.cuda(args.gpu)
    else:
        model.cuda()

    if args.opt_level is not None:
        logging.info("Using amp at opt level %s", args.opt_level)
        model = amp.initialize(model, opt_level=args.opt_level)

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
    _, val_ds = load_data(args, "train")
    test_ds = load_data(args, "test")
    logging.info("Batches in validation set: %s", len(val_ds) if val_ds is not None else "None")
    logging.info("Batches in test set: %s", len(test_ds))

    if args.steps_per_epoch is None:
        args.steps_per_epoch = len(val_ds)

    # get criterion
    train_criterion, eval_criterion = get_criterion(args)
    logging.info("Train criterion: %s", train_criterion.__class__.__name__)
    logging.info("Eval criterion: %s", eval_criterion.__class__.__name__)

    # setup ignite engines
    process_fn = TrainFunc.from_args(args, model, optimizer, train_criterion, device=args.gpu)
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
    Loss(eval_criterion).attach(val_eval, "loss")
    val_fmt = "Validation Results - Epoch: {epoch} Loss: {loss:.5f}"
    if args.threshold is not None:
        logging.info("Attaching thresholded metrics")

        def threshold(output):
            ypred, ytrue = output
            ypred = (ypred > args.threshold).byte()
            return ypred.flatten(), ytrue.flatten()

        Precision(output_transform=threshold).attach(val_eval, "precision")
        Recall(output_transform=threshold).attach(val_eval, "recall")
        ROC_AUC(output_transform=threshold).attach(val_eval, "roc_auc")
        val_fmt += " Precision: {precision:.5f}, Recall: {recall:.5f}, ROC_AUC: {roc_auc:.5f}"

    restore = CheckpointLoader.from_args(args, **to_save)

    if args.multiprocessing_distributed and args.rank % args.gpus_per_node != 0:
        pass
    else:
        summary = SummaryWriter.from_args(args, model, transform=lambda x: x.frames.rename(None))
        summary.attach(trainer)

        checkpoint = ModelCheckpoint.from_args(args)
        checkpoint.attach(trainer, args, **to_save)

        visualizer = OutputVisualizer.from_args(
            args, lambda x: preprocess_fn(x, args.threshold), visualize_fn, dpi=142
        )
        visualizer.attach(val_eval)

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
