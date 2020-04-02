#!/usr/bin/env python
# -*- coding: utf-8 -*-
from .flags import (
    add_checkpoint_args,
    add_classification_args,
    add_criterion_args,
    add_distributed_args,
    add_file_args,
    add_ignite_args,
    add_logging_args,
    add_lr_schedule_args,
    add_model_args,
    add_optimizer_args,
    add_preprocessing_args,
    add_regularization_args,
    add_runtime_args,
    add_tensorboard_args,
    add_training_args,
    add_validation_args,
    parse_args,
    setup_parser,
)
from .validators import NumericValidator
from .vision import add_vision_args
