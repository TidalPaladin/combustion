#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.nn as nn

from .focal_loss import FocalLoss, FocalLossWithLogits, focal_loss, focal_loss_with_logits
from .loss import WeightedBCEFromLogitsLoss, WeightedBCELoss, WeightedMSELoss, WeightedSoftMarginLoss, get_criterion
