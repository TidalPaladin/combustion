#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from combustion.lightning import HydraMixin
from combustion.lightning.metrics import BoxAveragePrecision
from combustion.models import EfficientDetFCOS as EffDetFCOS
from combustion.nn import FCOSLoss, FCOSDecoder
from combustion.vision import split_bbox_scores_class, split_box_target, to_8bit
from torchmetrics import MetricCollection


# from timm import create_model
from effdet import create_model

from ..mixins import VisualizationMixin


class EfficientDetFCOS(HydraMixin, pl.LightningModule, VisualizationMixin):
    def __init__(
        self,
        num_classes: int,
        compound_coeff: int,
        effdet_backbone: str = "tf_efficientdet_d4",
        strides: List[int] = [8, 16, 32, 64, 128],
        sizes: List[Tuple[int, int]] = [(-1, 64), (64, 128), (128, 256), (256, 512), (512, 10000000)],
        threshold: Optional[float] = None,
        nms_threshold: Optional[float] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.num_classes = int(num_classes)
        self.strides = [int(x) for x in strides]
        self.sizes = [(int(x), int(y)) for x, y in sizes]

        # TODO train this from scratch using combustion EfficientDet
        # self._model = EffDetFCOS.from_predefined(
        #    compound_coeff, self.num_classes, fpn_levels=[3, 5, 7, 8, 9], strides=self.strides
        # )

        self._model = create_model(effdet_backbone, pretrained=True)
        del self._model.box_net
        del self._model.class_net

        fpn_filters = self._model.config.fpn_channels
        num_repeats = 4

        self.fcos = FCOSDecoder(fpn_filters, self.num_classes, num_repeats, strides)

        self.threshold = float(threshold) if threshold is not None else 0.05
        self.nms_threshold = float(nms_threshold) if nms_threshold is not None else 0.1
        self._criterion = FCOSLoss(self.strides, self.num_classes, radius=1, interest_range=self.sizes)

        # metrics
        metrics = MetricCollection({
            f"ap{thresh}": BoxAveragePrecision(iou_threshold=thresh / 100, compute_on_step=True)
            for thresh in (25, 50, 75)
        })
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

        # freeze backbone
        for param in self._model.parameters():
            param.requires_grad = False

    def forward(self, inputs: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        height, width = inputs.shape[-2], inputs.shape[-1]
        # cls, reg, centerness = self._model.forward(inputs)
        features = self._model.backbone(inputs)
        features = self._model.fpn(features)
        cls, reg, centerness = self.fcos(features)
        return cls, reg, centerness

    def criterion(
        self,
        cls: Tensor,
        reg: Tensor,
        centerness: Tensor,
        target: Tensor,
        reg_scaling: float = 1.0,
        centerness_scaling: float = 1.0,
    ) -> Dict[str, Tensor]:
        # get losses
        target_bbox, target_cls = split_box_target(target)
        cls_loss, reg_loss, centerness_loss = self._criterion(cls, reg, centerness, target_bbox, target_cls)

        # get number of boxes for normalization
        num_boxes = (target_cls != -1).sum().clamp_min(1)
        cls_loss = cls_loss / num_boxes
        reg_loss = reg_loss / num_boxes
        centerness_loss = centerness_loss / num_boxes

        # compute a total loss
        total_loss = cls_loss + reg_loss * reg_scaling + centerness_loss * centerness_scaling

        return {
            "type_loss": cls_loss,
            "centerness_loss": centerness_loss,
            "reg_loss": reg_loss,
            "total_loss": total_loss,
        }

    def step(self, prefix: str, batch: Tuple[Tensor, Tensor], batch_nb: int):
        # assume already filtered based on only_classes
        img, target = batch
        self.last_input = img.clone()
        self.last_target = (img.clone(), {"coords": target[..., :4], "class": target[..., 4:]})

        target = target.long()

        # forward pass and loss
        cls, reg, centerness = self(img)
        loss_dict = self.criterion(cls, reg, centerness, target)

        # log images / metrics
        with torch.no_grad():
            self.last_cls = (img.clone().detach(), FCOSDecoder.reduce_heatmaps([x.sigmoid() for x in cls]))
            self.last_centerness = (img.clone().detach(), FCOSDecoder.reduce_heatmaps([x.sigmoid() for x in centerness]))

            # box generation can be slow so only do it at inference time
            if not self.training:
                pred_boxes = FCOSDecoder.postprocess(
                    cls,
                    reg,
                    centerness,
                    self._criterion.strides,
                    from_logits=True,
                    threshold=self.threshold,
                    nms_threshold=self.nms_threshold,
                    max_boxes=140,
                )
                bbox_coord, bbox_score, bbox_cls = pred_boxes.split((4, 1, 1), dim=-1)
                self.last_pred = (img, {"coords": bbox_coord, "score": bbox_score, "class": bbox_cls})

                # compute metrics
                metrics = getattr(self, f"{prefix}_metrics")
                for p, t in zip(pred_boxes, target):
                    metrics.update(p, t)
                self.log_dict(metrics.compute(), sync_dist=not self.training, on_step=self.training, on_epoch=not self.training)

        self.log_dict(
            {
                f"{prefix}/loss/total": loss_dict["total_loss"],
                f"{prefix}/loss/type": loss_dict["type_loss"],
                f"{prefix}/loss/centerness": loss_dict["centerness_loss"],
                f"{prefix}/loss/regression": loss_dict["reg_loss"],
            },
            sync_dist=not self.training,
            on_step=self.training,
            on_epoch=not self.training,
        )

        return loss_dict["total_loss"]

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_nb: int):
        lr = torch.as_tensor(self.get_lr())
        self.log("train/lr", lr, prog_bar=True)
        return self.step("train", batch, batch_nb)

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_nb: int):
        return self.step("val", batch, batch_nb)

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_nb: int):
        return self.step("test", batch, batch_nb)
