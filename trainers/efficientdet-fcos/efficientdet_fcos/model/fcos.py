#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import random
from typing import Optional, Tuple, List, Union, Dict

from torch.distributed import ReduceOp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.ops.boxes import nms as nms_torch
from torch.utils.checkpoint import checkpoint

from combustion.models import EfficientDetFCOS as EffDetFCOS
from combustion.vision import AnchorsToPoints, PointsToAnchors, to_8bit
from combustion.vision import nms as nms_combustion
from combustion.vision.anchors import ClipBoxes
from combustion.util import alpha_blend, apply_colormap
from medcog_preprocessing.data import Embeddings

from .mammogram_loss import MammogramLoss
import kornia
from kornia.augmentation import RandomCrop, RandomErasing
import pytorch_lightning as pl


from combustion.lightning import HydraMixin
from combustion.nn import AttentionUpsample2d, FCOSLoss
from combustion.vision import ConfusionMatrixIoU, BinaryLabelIoU
from combustion.vision.centernet import CenterNetMixin

from pytorch_lightning.metrics.classification import Accuracy, Fbeta


class EfficientDetFCOS(HydraMixin, pl.LightningModule):
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
        super(EfficientCenterDet, self).__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.num_classes = int(num_classes)
        self.strides = [int(x) for x in strides]
        self.sizes = [(int(x), int(y)) for x, y in sizes]
        self._model = EffDetFCOS.from_predefined(compound_coeff, self.num_classes, fpn_levels=[2, 3, 5, 7], strides=self.strides)

        out_channels = self._model.input_filters

        self.threshold = threshold
        self.nms_threshold = float(nms_threshold) if nms_threshold is not None else 0.1
        self._criterion = FCOSLoss(self.strides, self.num_classes, radius=1, interest_range=self.sizes)

        self.embeddings = Embeddings()
        self.embeddings.remap_types(keep_types)

    def update_bn(self, x: nn.Module):
        for name, child in x.named_children():
            if isinstance(child, nn.BatchNorm2d):
                setattr(x, name, nn.BatchNorm2d(child.num_features, track_running_stats=False))
            else:
                self.update_bn(child)

    def forward(self, inputs: Tensor) -> Tuple[Tensor, Tensor]:
        height, width = inputs.shape[-2], inputs.shape[-1]
        inputs = inputs[..., :self.in_channels, :, :]
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
        target_bbox, target_cls, target_malig = CenterNetMixin.split_box_target(target, split_label=True)
        cls_loss, reg_loss, centerness_loss = self._criterion(cls, reg, centerness, target_bbox, target_cls)

        # compute a total loss
        total_loss = cls_loss + reg_loss * reg_scaling + centerness_loss * centerness_scaling

        return {
            "type_loss": cls_loss,
            "centerness_loss": centerness_loss,
            "reg_loss": reg_loss,
            "total_loss": total_loss
        }

    def training_step(self, batch, batch_nb):
        # assume already filtered based on only_classes
        img, target = batch
        target = target.long()

        # forward pass
        cls, reg, centerness = self(img)

        loss_dict = self.criterion(cls, reg, centerness, target)

        # log training target heatmap on first step
        if self.trainer.global_step % 300 == 0:
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
            sync_dist=True,
        )

        return loss_dict["total_loss"]

    def _log_training_inputs(self, img, types, reg, centerness, target):
        step = self.trainer.global_step
        img_channel = img[..., :3, :, :]
        img_channel = self.apply_log_limit(img_channel)
        img_channel = self.apply_log_resolution(img_channel)
        img_channel = to_8bit(img_channel, same_on_batch=True)
        self.add_images(self.logger.experiment, f"train/input/channel_{i}", img_channel, step, split_batches=False, add_postfix=False)

        sizes = [x.shape[-2:] for x in types]
        bbox_target, cls_target, malig_target = CenterNetMixin.split_box_target(target, split_label=True)
        cls_heatmap, reg_heatmap, centerness_heatmap = self._criterion.create_targets(bbox_target, cls_target[..., 0:1], sizes)

        # overlay target boxes onto image for visualization
        img_with_boxes = img[..., 0:1, :, :].clone()
        scale = self.get_resolution_scale_factor(img_with_boxes)
        img_with_boxes = F.interpolate(img_with_boxes, scale_factor=scale, mode="nearest")
        bbox_target = bbox_target.clone().float()
        bbox_target = bbox_target[..., :4].mul_(scale)
        img_with_boxes = self.overlay_boxes(
            img_with_boxes, 
            bbox_target, 
            cls_target, 
            class_names=self.embeddings.reverse_type_embeddings,
            box_color=(0, 255, 0),
            text_color=(0, 0, 0)
        )

        self.log_heatmap(
            self.logger.experiment, 
            "train/heatmap/pred/", 
            img_with_boxes,
            EfficientDetFCOS.reduce_heatmap([torch.sigmoid(x) for x in types]), 
            step, 
            self.embeddings.reverse_type_embeddings,
            image_alpha=1.0,
            heatmap_alpha=0.7,
            split_batches=False
        )

        pred_boxes, _ = EfficientDetFCOS.create_boxes(
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
            class_names=self.embeddings.reverse_type_embeddings,
        )
        self.add_images(self.logger.experiment, "train/pred", viz, step, split_batches=False)

        reg_heatmap = [torch.norm(x.float().clamp_min_(0), p=2, dim=-3, keepdim=True) for x in reg_heatmap]
        reg_heatmap = [x.div(x.amax(dim=(-1, -2, -3), keepdim=True).clamp_min(1)) for x in reg_heatmap]

        hm =  EfficientDetFCOS.reduce_heatmap(reg_heatmap)
        hm = F.interpolate(hm, scale_factor=scale, mode="nearest")
        self.log_heatmap(
            self.logger.experiment, 
            "train/reg/true/", 
            img_with_boxes,
            hm,
            step, 
            self.embeddings.reverse_type_embeddings,
            image_alpha=1.0,
            heatmap_alpha=0.7,
            split_batches=False
        )

        hm =  EfficientDetFCOS.reduce_heatmap(cls_heatmap)
        hm = F.interpolate(hm, scale_factor=scale, mode="nearest")
        self.log_heatmap(
            self.logger.experiment, 
            "train/heatmap/true/", 
            img_with_boxes,
            hm,
            step, 
            self.embeddings.reverse_type_embeddings,
            image_alpha=1.0,
            heatmap_alpha=0.7,
            split_batches=False
        )
        hm = EfficientDetFCOS.reduce_heatmap([x.clamp_min(0) for x in centerness_heatmap])
        hm = F.interpolate(hm, scale_factor=scale, mode="nearest")
        self.log_heatmap(
            self.logger.experiment, 
            "train/centerness/true/", 
            img_with_boxes,
            hm,
            step, 
            self.embeddings.reverse_type_embeddings,
            image_alpha=1.0,
            heatmap_alpha=0.7,
            split_batches=False
        )

        self.add_images(self.logger.experiment, "train/true", img_with_boxes, step, split_batches=False)

    def _log_training_metrics(self, types, centerness, target):

        tar_bbox, tar_type = self.split_box_target(target)

        # postprocess predicted heatmaps
        #type_heatmap, malig_heatmap = self.postprocess(type_heatmap, malig_heatmap)

        # compute pred target pairs for metric
        #global_hm = EfficientDetFCOS.reduce_heatmap([x.sigmoid() * y.sigmoid().expand_as(x) for x, y in zip(types, centerness)])
        global_hm = EfficientDetFCOS.reduce_heatmap([x.sigmoid() for x in types])
        global_type_pairs, global_malig_pairs = self.get_global_pred_target_pairs_on_batch(
            global_hm,
            None, 
            tar_bbox, 
            tar_type, 
            None
        )

        # compute local and global metrics using pred target pairs
        global_metrics = self.compute_global_metrics(global_type_pairs, global_type_pairs)
        #local_metrics = self.compute_local_metrics(local_type_pairs, local_malig_pairs, num_classes=self.num_classes)

        # log non-scalar metrics and get a dict of scalar metrics that need to be logged
        metric_dict = self.log_metrics(
            self.logger.experiment, 
            "train_", 
            global_metrics, 
            None, 
            self.embeddings.reverse_type_embeddings, 
            self.trainer.global_step
        )
        self.log_dict(
            metric_dict, 
            sync_dist=True,
            on_epoch=False,
            on_step=True
        )

    def on_validation_epoch_start(self):
        self.reset_counters()

    def validation_step(self, batch, batch_nb):
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
                #"val/loss/malignancy": loss_dict["malig_loss"],
            },
            sync_dist=True,
            reduce_fx=lambda x: x.mean(),
            on_epoch=True,
            on_step=False
        )

        batch_size = img.shape[0]

        # convert bounding box target to heatmap (assuming malignancy is at last index)
        tar_bbox, tar_type, tar_malig = self.split_box_target(target, split_label=True)

        sizes = [x.shape[-2:] for x in cls]
        target_cls_heatmap, target_reg_heatmap, target_centerness_heatmap = self._criterion.create_targets(tar_bbox, tar_type, sizes)
        #target_malig_heatmap = self.create_malignancy_heatmap(self.anchorsToPoints, tar_bbox, tar_type, tar_malig, img.shape[-2:])

        # turn postprocessed heatmaps into predicted boxes
        # x1, y1, x2, y2, type_score, malig_score, type_class
        pred_boxes, _ = EfficientDetFCOS.create_boxes(
            cls,
            reg,
            centerness,
            self._criterion.strides,
            from_logits=True,
            threshold=self.threshold,
            nms_threshold=self.nms_threshold,
        )

        # visualization
        self.visualization_step("val_", img, cls, pred_boxes, target, self.current_epoch)

        # compute pred target pairs for metric
        #global_hm = EfficientDetFCOS.reduce_heatmap([x.sigmoid() * y.sigmoid().expand_as(x) for x, y in zip(cls, centerness)])
        global_hm = EfficientDetFCOS.reduce_heatmap([x.sigmoid() for x in cls])
        global_type_pairs, global_malig_pairs = self.get_global_pred_target_pairs_on_batch(
            global_hm,
            None, 
            tar_bbox, 
            tar_type, 
            None
        )

        #local_type_pairs, local_malig_pairs = self.get_local_pred_target_pairs_on_batch(
        #    type_heatmap, 
        #    malig_heatmap, 
        #    tar_bbox, 
        #    tar_type, 
        #    tar_malig,
        #    upsample=self.downsample
        #)

        result = {
            "global_type_pairs": global_type_pairs,
            #"global_malig_pairs": global_malig_pairs,
            #"local_type_pairs": local_type_pairs,
            #"local_malig_pairs": local_malig_pairs,
            #"val/loss/total": loss_dict["total_loss"],
        }


        # increment counters for image logging
        self.increment_counters(batch_size)

        return result
#
    def visualization_step(self, prefix, img, types, pred, target, step):
        # drop sobel channel if present for visualization
        img = img[..., 0:1, :, :]

        # split some of the extracted tensors
        pred_bbox, pred_scores, pred_type = self.split_bbox_scores_class(pred)
        tar_bbox, tar_type, tar_malig = self.split_box_target(target, split_label=True)

        # overlay target boxes onto image for visualization
        img_with_boxes = img[..., 0:1, :, :].clone()
        scale = self.get_resolution_scale_factor(img_with_boxes)
        img_with_boxes = F.interpolate(img_with_boxes, scale_factor=scale, mode="nearest")
        tar_bbox, pred_bbox = tar_bbox.clone().float(), pred_bbox.clone()
        tar_bbox[..., :4].mul_(scale)
        pred_bbox[..., :4].mul_(scale)
        img_with_boxes = self.overlay_boxes(
            img_with_boxes, 
            tar_bbox, 
            tar_type, 
            class_names=self.embeddings.reverse_type_embeddings,
            box_color=(0, 255, 0),
            text_color=(0, 0, 0)
        )
        # overlay predicted boxes onto image for visualization
        viz = self.overlay_boxes(
            img_with_boxes.clone(), 
            pred_bbox, 
            pred_type, 
            pred_scores, 
            class_names=self.embeddings.reverse_type_embeddings
        )
        viz = self.apply_log_limit(viz)
        self.add_images(self.logger.experiment, f"{prefix}boxes", viz, step)


        # TODO separate malignancy-only boxes visualization

        sizes = [x.shape[-2:] for x in types]

        hm = EfficientDetFCOS.reduce_heatmap([torch.sigmoid(x.detach()) for x in types])
        hm = F.interpolate(hm, scale_factor=scale, mode="nearest")
        self.log_heatmap(
            self.logger.experiment, 
            f"{prefix}/heatmap/pred/", 
            img_with_boxes,
            hm,
            step, 
            self.embeddings.reverse_type_embeddings,
            image_alpha=1.0,
            heatmap_alpha=0.7,
        )


        # overlay target boxes onto image for visualization
        viz = self.overlay_boxes(
            img_with_boxes.clone(), 
            pred_bbox, 
            pred_type, 
            pred_scores, 
            class_names=self.embeddings.reverse_type_embeddings,
        )
        self.add_images(self.logger.experiment, f"{prefix}boxes", img_with_boxes.clone(), step)




    def validation_epoch_end(self, result):
        # process per-epoch pred target pairs
        global_type_pairs = torch.cat([x["global_type_pairs"] for x in result], dim=0)
        #global_malig_pairs = torch.cat([x["global_malig_pairs"] for x in result], dim=0)
        #local_type_pairs = [y for x in result for y in x["local_type_pairs"]]
        #local_malig_pairs = [y for x in result for y in x["local_malig_pairs"]]
        #loss = torch.stack([x["val/loss/total"] for x in result], dim=0).mean()

        # compute local and global metrics using pred target pairs
        global_metrics = self.compute_global_metrics(global_type_pairs, global_type_pairs)
        #local_metrics = self.compute_local_metrics(local_type_pairs, local_malig_pairs, num_classes=self.num_classes)

        # log non-scalar metrics and get a dict of scalar metrics that need to be logged
        metric_dict = self.log_metrics(
            self.logger.experiment, 
            "val_", 
            global_metrics, 
            None, 
            self.embeddings.reverse_type_embeddings, 
            self.trainer.global_step
        )
        self.log_dict(metric_dict, sync_dist=True)
        #self.logger.log_hyperparams(self.hparams, metrics={"val/loss/total": loss})

    def on_test_epoch_start(self):
        self.reset_counters()

    # TODO this is basically copy paste from validation_step
    def test_step(self, batch, batch_nb):
        # read in image and bounding box target
        # assume already filtered based on only_classes
        img, target = batch
        batch_size = img.shape[0]

        # convert bounding box target to heatmap (assuming malignancy is at last index)
        tar_bbox, tar_type, tar_malig = self.split_box_target(target, split_label=True)
        target_type_heatmap = self.anchorsToPoints(tar_bbox, tar_type, shape=img.shape[-2:])
        target_malig_heatmap = self.create_malignancy_heatmap(self.anchorsToPoints, tar_bbox, tar_type, tar_malig, img.shape[-2:])

        # forward pass to get predicted heatmaps
        type_heatmap, malig_heatmap = self(img)

        # compute loss
        loss_dict = self.criterion(type_heatmap, malig_heatmap, target_type_heatmap, target_malig_heatmap)
        self.log_dict(
            {
                "test/loss/total": loss_dict["total_loss"],
                "test/loss/type": loss_dict["type_loss"],
                "test/loss/malignancy": loss_dict["malig_loss"],
                "test/loss/regression": loss_dict["reg_loss"],
            },
            sync_dist=True,
            reduce_fx=lambda x: x.mean(),
            on_epoch=True,
            on_step=False
        )

        # postprocess predicted heatmaps
        type_heatmap, malig_heatmap = self.postprocess(type_heatmap, malig_heatmap)

        # turn postprocessed heatmaps into predicted boxes
        # x1, y1, x2, y2, type_score, malig_score, type_class
        pred = self.create_boxes(img, type_heatmap, malig_heatmap)

        # compute pred target pairs for metric
        global_type_pairs, global_malig_pairs = self.get_global_pred_target_pairs_on_batch(
            type_heatmap, 
            malig_heatmap, 
            tar_bbox, 
            tar_type, 
            tar_malig
        )
        local_type_pairs, local_malig_pairs = self.get_local_pred_target_pairs_on_batch(
            type_heatmap, 
            malig_heatmap, 
            tar_bbox, 
            tar_type, 
            tar_malig,
            upsample=self.downsample
        )
        result = {
            "global_type_pairs": global_type_pairs,
            "global_malig_pairs": global_malig_pairs,
            "local_type_pairs": local_type_pairs,
            "local_malig_pairs": local_malig_pairs,
            #"test/loss/total": loss_dict["total_loss"],
        }

        # visualization
        self.visualization_step("test_", img, type_heatmap, pred, target, self.current_epoch)

        # increment counters for image logging
        self.increment_counters(batch_size)

        return result

    def test_epoch_end(self, result):
        # process per-epoch pred target pairs
        global_type_pairs = torch.cat([x["global_type_pairs"] for x in result], dim=0)
        global_malig_pairs = torch.cat([x["global_malig_pairs"] for x in result], dim=0)
        local_type_pairs = [y for x in result for y in x["local_type_pairs"]]
        local_malig_pairs = [y for x in result for y in x["local_malig_pairs"]]
        #loss = torch.stack([x["test/loss/total"] for x in result], dim=0).mean()

        # compute local and global metrics using pred target pairs
        global_metrics = self.compute_global_metrics(global_type_pairs, global_malig_pairs)
        local_metrics = self.compute_local_metrics(local_type_pairs, local_malig_pairs, num_classes=self.num_classes)

        # log non-scalar metrics and get a dict of scalar metrics that need to be logged
        metric_dict = self.log_metrics(
            self.logger.experiment, 
            "test_", 
            global_metrics, 
            local_metrics, 
            self.embeddings.reverse_type_embeddings, 
            self.trainer.global_step
        )
        self.log_dict(metric_dict, sync_dist=True)

