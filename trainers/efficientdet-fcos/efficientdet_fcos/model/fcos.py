#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from combustion.lightning import HydraMixin
from combustion.lightning.metrics import BoxAveragePrecision
from combustion.models import EfficientDetFCOS as EffDetFCOS
from combustion.nn import FCOSLoss
from combustion.vision import split_bbox_scores_class, split_box_target, to_8bit

from ..mixins import VisualizationMixin


class EfficientDetFCOS(HydraMixin, pl.LightningModule, VisualizationMixin):
    def __init__(
        self,
        num_classes: int,
        compound_coeff: int,
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
        self._model = EffDetFCOS.from_predefined(
            compound_coeff, self.num_classes, fpn_levels=[3, 5, 7], strides=self.strides
        )

        self._model.input_filters

        self.threshold = threshold
        self.nms_threshold = float(nms_threshold) if nms_threshold is not None else 0.1
        self._criterion = FCOSLoss(self.strides, self.num_classes, radius=1, interest_range=self.sizes)

        # metrics
        self.ap25, self.ap50, self.ap75 = [
            BoxAveragePrecision(iou_threshold=thresh, compute_on_step=False, dist_sync_on_step=False)
            for thresh in (0.25, 0.5, 0.75)
        ]

    def update_bn(self, x: nn.Module):
        for name, child in x.named_children():
            if isinstance(child, nn.BatchNorm2d):
                setattr(x, name, nn.BatchNorm2d(child.num_features, track_running_stats=False))
            else:
                self.update_bn(child)

    def forward(self, inputs: Tensor) -> Tuple[Tensor, Tensor]:
        height, width = inputs.shape[-2], inputs.shape[-1]
        cls, reg, centerness = self._model.forward(inputs)
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
        # match segmentation size with target

        # get losses
        target_bbox, target_cls = split_box_target(target)
        cls_loss, reg_loss, centerness_loss = self._criterion(cls, reg, centerness, target_bbox, target_cls)

        # compute a total loss
        total_loss = cls_loss + reg_loss * reg_scaling + centerness_loss * centerness_scaling

        return {
            "type_loss": cls_loss,
            "centerness_loss": centerness_loss,
            "reg_loss": reg_loss,
            "total_loss": total_loss,
        }

    def training_step(self, batch, batch_nb):
        # assume already filtered based on only_classes
        img, target = batch
        target = target.long()

        # forward pass
        cls, reg, centerness = self(img)

        loss_dict = self.criterion(cls, reg, centerness, target)

        # log training target heatmap on first step
        if self.trainer.global_step % 1000 == 0:
            with torch.no_grad():
                self._log_training_inputs(img, cls, reg, centerness, target)
                self.reset_counters()

        lr = torch.as_tensor(self.get_lr())
        self.log("train/lr", lr, prog_bar=True)
        self.log_dict(
            {
                "train/loss/total": loss_dict["total_loss"],
                "train/loss/type": loss_dict["type_loss"],
                "train/loss/centerness": loss_dict["centerness_loss"],
                "train/loss/regression": loss_dict["reg_loss"],
            },
            sync_dist=False,
        )

        return loss_dict["total_loss"]

    def _log_training_inputs(self, img, types, reg, centerness, target):
        step = self.trainer.global_step
        img_channel = img[..., :3, :, :]
        img_channel = self.apply_log_limit(img_channel)
        img_channel = self.apply_log_resolution(img_channel)
        img_channel = to_8bit(img_channel, same_on_batch=True)
        self.add_images(
            self.logger.experiment, "train/input", img_channel, step, split_batches=False, add_postfix=False
        )

        sizes = [x.shape[-2:] for x in types]
        bbox_target, cls_target = split_box_target(target)
        cls_heatmap, reg_heatmap, centerness_heatmap = self._criterion.create_targets(bbox_target, cls_target, sizes)

        # overlay target boxes onto image for visualization
        img_with_boxes = img.clone()
        scale = self.get_resolution_scale_factor(img_with_boxes)
        img_with_boxes = F.interpolate(img_with_boxes, scale_factor=scale, mode="nearest")
        bbox_target = bbox_target.clone().float()
        bbox_target = bbox_target[..., :4].mul_(scale)
        img_with_boxes = self.overlay_boxes(
            img_with_boxes,
            bbox_target,
            cls_target,
            # class_names=COCO_EMBEDDINGS,
            box_color=(0, 255, 0),
            text_color=(0, 0, 0),
        )

        self.log_heatmap(
            self.logger.experiment,
            "train/heatmap/pred/",
            img_with_boxes,
            EffDetFCOS.reduce_heatmap([torch.sigmoid(x) for x in types]),
            step,
            # COCO_EMBEDDINGS,
            image_alpha=1.0,
            heatmap_alpha=0.7,
            split_batches=False,
        )

        pred_boxes, _ = EffDetFCOS.create_boxes(
            types,
            reg,
            centerness,
            self._criterion.strides,
            from_logits=True,
            threshold=self.threshold,
            nms_threshold=self.nms_threshold,
        )
        pred_boxes[..., :4].mul_(scale)

        # overlay target boxes onto image for visualization
        viz = self.overlay_boxes(
            img_with_boxes.clone(),
            pred_boxes[..., :4],
            pred_boxes[..., 5:6],
            pred_boxes[..., 4:5],
            # class_names=COCO_EMBEDDINGS,
        )
        self.add_images(self.logger.experiment, "train/pred", viz, step, split_batches=False)

        reg_heatmap = [torch.norm(x.float().clamp_min_(0), p=2, dim=-3, keepdim=True) for x in reg_heatmap]
        reg_heatmap = [x.div(x.amax(dim=(-1, -2, -3), keepdim=True).clamp_min(1)) for x in reg_heatmap]

        hm = EffDetFCOS.reduce_heatmap(reg_heatmap)
        hm = F.interpolate(hm, scale_factor=scale, mode="nearest")
        self.log_heatmap(
            self.logger.experiment,
            "train/reg/true/",
            img_with_boxes,
            hm,
            step,
            # COCO_EMBEDDINGS,
            image_alpha=1.0,
            heatmap_alpha=0.7,
            split_batches=False,
        )

        hm = EffDetFCOS.reduce_heatmap(cls_heatmap)
        hm = F.interpolate(hm, scale_factor=scale, mode="nearest")
        self.log_heatmap(
            self.logger.experiment,
            "train/heatmap/true/",
            img_with_boxes,
            hm,
            step,
            # COCO_EMBEDDINGS,
            image_alpha=1.0,
            heatmap_alpha=0.7,
            split_batches=False,
        )
        hm = EffDetFCOS.reduce_heatmap([x.clamp_min(0) for x in centerness_heatmap])
        hm = F.interpolate(hm, scale_factor=scale, mode="nearest")
        self.log_heatmap(
            self.logger.experiment,
            "train/centerness/true/",
            img_with_boxes,
            hm,
            step,
            # COCO_EMBEDDINGS,
            image_alpha=1.0,
            heatmap_alpha=0.7,
            split_batches=False,
        )

        self.add_images(self.logger.experiment, "train/true", img_with_boxes, step, split_batches=False)

    def on_validation_epoch_start(self):
        self.reset_counters()

    def validation_step(self, batch, batch_idx):
        img, target = batch
        target = target.long()

        # forward pass
        cls, reg, centerness = self(img)

        loss_dict = self.criterion(cls, reg, centerness, target)

        self.log_dict(
            {
                "val/loss/total": loss_dict["total_loss"],
                "val/loss/type": loss_dict["type_loss"],
                "val/loss/centerness": loss_dict["centerness_loss"],
                "val/loss/regression": loss_dict["reg_loss"],
            },
            sync_dist=False,
            reduce_fx=lambda x: x.mean(),
            on_epoch=True,
            on_step=False,
        )

        batch_size = img.shape[0]

        tar_bbox, tar_type = split_box_target(target)

        sizes = [x.shape[-2:] for x in cls]
        target_cls_heatmap, target_reg_heatmap, target_centerness_heatmap = self._criterion.create_targets(
            tar_bbox, tar_type, sizes
        )

        # turn postprocessed heatmaps into predicted boxes
        # x1, y1, x2, y2, type_score, type_class
        pred_boxes, _ = EffDetFCOS.create_boxes(
            cls,
            reg,
            centerness,
            self._criterion.strides,
            from_logits=True,
            threshold=self.threshold,
            nms_threshold=self.nms_threshold,
        )

        # compute metrics
        for metric in (self.ap25, self.ap50, self.ap75):
            for p, t in zip(pred_boxes, target):
                metric.update(p, t)

        # visualization
        if batch_idx < 32:
            self.visualization_step("val_", img, cls, pred_boxes, target, self.current_epoch)
            self.increment_counters(batch_size)

    def visualization_step(self, prefix, img, types, pred, target, step):
        # split some of the extracted tensors
        pred_bbox, pred_scores, pred_type = split_bbox_scores_class(pred)
        tar_bbox, tar_type = split_box_target(target)

        # overlay target boxes onto image for visualization
        img_with_boxes = img.clone()
        scale = self.get_resolution_scale_factor(img_with_boxes)
        img_with_boxes = F.interpolate(img_with_boxes, scale_factor=scale, mode="nearest")
        tar_bbox, pred_bbox = tar_bbox.clone().float(), pred_bbox.clone()
        tar_bbox[..., :4].mul_(scale)
        pred_bbox[..., :4].mul_(scale)
        img_with_boxes = self.overlay_boxes(
            img_with_boxes,
            tar_bbox,
            tar_type,
            # class_names=COCO_EMBEDDINGS,
            box_color=(0, 255, 0),
            text_color=(0, 0, 0),
        )
        # overlay predicted boxes onto image for visualization
        viz = self.overlay_boxes(
            img_with_boxes.clone(),
            pred_bbox,
            pred_type,
            pred_scores,
            # class_names=COCO_EMBEDDINGS
        )
        viz = self.apply_log_limit(viz)
        self.add_images(self.logger.experiment, f"{prefix}boxes", viz, step)

        sizes = [x.shape[-2:] for x in types]

        hm = EffDetFCOS.reduce_heatmap([torch.sigmoid(x.detach()) for x in types])
        hm = F.interpolate(hm, scale_factor=scale, mode="nearest")
        self.log_heatmap(
            self.logger.experiment,
            f"{prefix}/heatmap/pred/",
            img_with_boxes,
            hm,
            step,
            # COCO_EMBEDDINGS,
            image_alpha=1.0,
            heatmap_alpha=0.7,
        )

        # overlay target boxes onto image for visualization
        viz = self.overlay_boxes(
            img_with_boxes.clone(),
            pred_bbox,
            pred_type,
            pred_scores,
            # class_names=COCO_EMBEDDINGS,
        )
        self.add_images(self.logger.experiment, f"{prefix}boxes", img_with_boxes.clone(), step)

    def validation_epoch_end(self, result):
        # compute metrics
        self.log("val/ap25", self.ap25, on_step=False, on_epoch=True)
        self.log("val/ap50", self.ap50, on_step=False, on_epoch=True)
        self.log("val/ap75", self.ap75, on_step=False, on_epoch=True)

    def on_test_epoch_start(self):
        self.reset_counters()
