#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Dict, Iterable, Union, Tuple
import pytorch_lightning as pl
from combustion.util import alpha_blend, apply_colormap, check_is_tensor, check_ndim_match
from combustion.vision import to_8bit, visualize_bbox
from torch.utils.tensorboard import SummaryWriter

class VisualizationMixin:

    __image_log_limit: int = 16
    __image_log_resolution: int = 1024
    split_batches: bool = True

    __batch_counter: int = 0
    __example_counter: int = 0

    def reset_counters(self):
        self.__batch_counter = 0
        self.__example_counter = 0

    def increment_counters(self, example: int, batch: int = 1):
        self.__batch_counter += batch
        self.__example_counter += example

    @property
    def global_batch_count(self):
        if hasattr(self, "global_rank"):
            return self.__batch_counter * (self.global_rank + 1)
        else:
            return self.__batch_counter 

    @property
    def global_example_count(self):
        if hasattr(self, "global_rank"):
            return self.__example_counter * (self.global_rank + 1)
        else:
            return self.__example_counter 

    @property
    def image_log_limit(self):
        return self.__image_log_limit

    @image_log_limit.setter
    def image_log_limit(self, val: int):
        self.__image_log_limit = int(val)

    def apply_log_limit(self, x: Tensor) -> Tensor:
        if self.image_log_limit and x.ndim == 4:
            limit = self.image_log_limit
            x = x[:limit, ...]
        return x

    @property
    def image_log_resolution(self):
        return self.__image_log_resolution

    @image_log_resolution.setter
    def image_log_resolution(self, val: int):
        self.__image_log_resolution = int(val)

    def apply_log_resolution(self, img: Tensor, bbox: Optional[Tensor] = None) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        height, width = img.shape[-2:]
        if height <= self.image_log_resolution and width <= self.image_log_resolution:
            if bbox is not None:
                return img, bbox
            else:
                return img

        max_dim = max(height, width)
        scale_factor = self.image_log_resolution / max_dim
        img = F.interpolate(img, scale_factor=scale_factor, mode="bilinear")
        if bbox is not None:
            bbox = bbox.clone()
            bbox[..., :4].floor_divide_(scale_factor)
            return img, bbox
        else:
            return img

    def get_resolution_scale_factor(self, img: Tensor) -> float:
        height, width = img.shape[-2:]
        if height <= self.image_log_resolution and width <= self.image_log_resolution:
            return 1
        max_dim = max(height, width)
        return self.image_log_resolution / max_dim

    @staticmethod
    def overlay_heatmap(image: Tensor, heatmap: Tensor, image_alpha: float = 1.0, heatmap_alpha: float = 0.5) -> Tensor:
        r"""Given a background image and per-class heatmaps, produce a visualization
        with each class' heatmap projected onto the background image.

        Args:
            image (:class:`torch.Tensor`):
                The background image

            heatmap (:class:`torch.Tensor`):
                The classification heatmap (with regression channels removed). If ``heatmap`` has a different
                size than ``image`` it will be adjusted to match.

            image_alpha (float):
                Alpha value for the background image when blending with heatmap

            heatmap_alpha (float):
                Alpha value for the heatmap when blending with background image

        Shape:
            * ``image`` - :math:`(*, 1, H, W)`
            * ``heatmap`` - :math:`(*, C, H', W')`
            * Output - :math:`(C, *, 3, H, W)`
        """
        check_is_tensor(image, "image")
        check_is_tensor(heatmap, "heatmap")
        check_ndim_match(image, heatmap, "image", "heatmap")

        # TODO enable these once methods are under test in combustion
        #if image.ndim > 3:
        #    check_shape(image, (None, 1, None, None), "image")
        #else:
        #    check_shape(image, (1, None, None), "image")

        # expand grayscale image channel to 3 grayscale channels
        if image.shape[-3] != 3:
            expansion = [1] * image.ndim
            expansion[-3] = 3
            image = image.repeat(*expansion)

        # convert image to 8-bit, then to range [0, 1]
        if image.dtype != torch.uint8:
            image = to_8bit(image, per_channel=False, same_on_batch=True)
        image = image.float().div_(255)

        # heatmap should have been passed through sigmoid and be in range [0, 1]
        assert heatmap.max() <= 1, f"max={heatmap.max()}"
        assert heatmap.min() >= 0, f"min={heatmap.min()}"

        # image and heatmap must both be on CPU
        heatmap = heatmap.clone()
        image = image.clone()
        #heatmap = heatmap.cpu()
        #image = image.cpu()

        # apply_colormap and alpha_blend expect a batch dim
        if heatmap.ndim == 3:
            heatmap.unsqueeze_(0)
        if image.ndim == 3:
            image.unsqueeze_(0)

        # loop over each class in the heatmap
        result = []
        for class_idx in range(heatmap.shape[-3]):
            heatmap_channel = heatmap[..., class_idx:class_idx+1, :, :]

            # apply colormap to heatmap channel
            heatmap_channel = apply_colormap(heatmap_channel)[:, :3, :, :]

            # upsample heatmap to match background image shape
            output_size = image.shape[-2:]
            heatmap_channel = F.interpolate(heatmap_channel, size=output_size, mode="nearest")

            # alpha blend channels together and return to 8-bit image
            assert image.shape[-3] == 3
            assert heatmap_channel.shape[-2:] == image.shape[-2:]
            heatmap_channel, _ = alpha_blend(heatmap_channel, image, heatmap_alpha, image_alpha)
            heatmap_channel = to_8bit(heatmap_channel, same_on_batch=True, per_channel=False)
            result.append(heatmap_channel)

        return torch.stack(result, dim=0)

    @staticmethod
    def overlay_boxes(
        image: Tensor, 
        bbox: Tensor, 
        classes: Tensor, 
        scores: Optional[Tensor] = None,
        class_names: Optional[dict] = None, 
        pad_value: float = -1, 
        **kwargs
    ) -> Tensor:
        r"""

        Shapes:
            * ``image`` - :math:`(*, 1, H, W)`
            * ``bbox`` - :math:`(*, N, 4)`
            * ``classes`` - :math:`(*, N, 1)`
            * ``scores`` - :math:`(*, N, 1)`
            * Output - :math:`(*, 3, H, W)`
        """
        check_is_tensor(image, "image")
        check_is_tensor(bbox, "bbox")
        check_is_tensor(classes, "classes")
        if scores is not None:
            check_is_tensor(scores, "scores")

        # TODO enable these once methods are under test in combustion
        # validate input shapes
        #if image.ndim == 4: 
        #    check_shape(image, (None, 1, None, None), "image")
        #    check_shape(bbox, (None, None, 4), "image")
        #    check_shape(classes, (None, None, 1), "classes")
        #    if scores is not None:
        #        check_shape(scores, (None, None, 1), "scores")
        #else:
        #    check_shape(image, (1, None, None), "image")
        #    check_shape(bbox, (None, 4), "image")
        #    check_shape(classes, (None, 1), "classes")
        #    if scores is not None:
        #        check_shape(scores, (None, 1), "scores")

        image = image.clone()

        # convert image to 8-bit
        if image.dtype != torch.uint8:
            image = to_8bit(image, same_on_batch=True, per_channel=False)

        # convert back to float in range [0, 1] for cv2
        if not image.is_floating_point():
            image = image.float().div_(255)

        # add 3 grayscale channels if input is single channel grayscale
        if image.shape[-3] == 1:
            expansion = [1] * image.ndim
            expansion[-3] = 3
            image = image.repeat(*expansion)

        # add batch dim if not present
        batched = bbox.ndim == 3
        if not batched:
            bbox = bbox.view(1, *bbox.shape)
            classes = classes.view(1, *classes.shape)
            if scores is not None:
                scores = scores.view(1, *scores.shape)
            image = image.view(1, *image.shape)

        assert image.ndim == 4
        assert classes.ndim == 3
        assert bbox.ndim == 3
        assert scores is None or scores.ndim == 3

        # add boxes to each image in the batch
        batch_size = bbox.shape[0]
        result = []
        valid_indices = ~((classes == pad_value).all(dim=-1))
        for batch_idx in range(batch_size):

            # select a batch and drop padding
            image_i = image[batch_idx]
            valid_indices_i = valid_indices[batch_idx]
            bbox_i = bbox[batch_idx][valid_indices_i]
            classes_i = classes[batch_idx][valid_indices_i]
            if scores is not None:
                scores_i = scores[batch_idx][valid_indices_i]
            else:
                scores_i = None

            # round and convert to long if float
            if bbox_i.is_floating_point():
                bbox_i = bbox_i.round().long()
            if classes_i.is_floating_point():
                classes_i = classes_i.round().long()


            result_i = visualize_bbox(
                image_i,
                bbox_i,
                classes_i,
                scores_i,
                class_names=class_names,
                thickness=1,
                **kwargs
            )
            result.append(result_i)

        result = torch.stack(result, dim=0)
        result = to_8bit(result, per_channel=False, same_on_batch=True)
        if not batched:
            result.squeeze_(0)
        return result

    def log_heatmap(
        self,
        logger: SummaryWriter, 
        prefix: str, 
        image: Tensor, 
        heatmap: Tensor, 
        step: int = 0, 
        class_dict: Optional[Dict[int, str]] = None,
        split_batches: Optional[bool] = None,
        **kwargs
    ) -> None:
        r"""Logs a heatmap to a TensorBoard experiment

        Args:
            logger (:class:`SummaryWriter`):
                Experiment to log to

            prefix (str):
                Prefix for each heatmap. The final tensorboard tag will be ``f"{prefix}{class_index}"``

            image (:class:`torch.Tensor`):
                Background image for the heatmap overlay

            heatmap (:class:`torch.Tensor`):
                Heatmap to log. If ``heatmap`` has a different size than ``image`` it will be 
                adjusted to match.

            step (int):
                TensorBoard step value

            class_dict (optional, dict):
                Dictionary mapping integer class values to textual names

            **kwargs:
                Forwarded to :func:`VisualizationMixin.overlay_heatmap`

        Shape:
            * ``image`` - :math:`(*, 1, H, W)`
            * ``heatmap`` - :math:`(*, C, H', W')` where :math:`C` is the number of classes
        """
        if not isinstance(heatmap, Tensor) and isinstance(heatmap, Iterable):
            for i, h in enumerate(heatmap):
                new_prefix = f"{prefix}/level_{i}/"
                self.log_heatmap(logger, new_prefix, image, h, step, class_dict, split_batches, **kwargs)
            return

        check_is_tensor(image, "image")
        check_is_tensor(heatmap, "heatmap")

        # get images to display
        # shape = (C, *, 3, H, W)
        out_image = VisualizationMixin.overlay_heatmap(image, heatmap, **kwargs)
        assert out_image.shape[0] == heatmap.shape[-3]
        assert out_image.shape[-2:] == image.shape[-2:]
        assert out_image.shape[-3] == 3

        for i, class_result in enumerate(out_image):
            cls_name = i if class_dict is None else class_dict[i]
            tag = f"{prefix}{cls_name}" 
            self.add_images(logger, tag, class_result, step, split_batches=split_batches)

    def add_images(self, logger: SummaryWriter, tag: str, x: Tensor, step: int = 0, split_batches: Optional[bool] = None, add_postfix: bool = True):
        split_batches = self.split_batches if split_batches is None else split_batches

        if x.ndim == 4:
            if split_batches:
                for frame in x:
                    subtag = tag if not add_postfix else f"{tag}/example_{self.global_example_count}"
                    logger.add_image(subtag, frame, step)
            else:
                subtag = tag if not add_postfix else f"{tag}/batch_{self.global_batch_count}"
                logger.add_images(subtag, x, step)
        else:
            subtag = tag if not add_postfix else f"{tag}/example_{self.global_example_count}"
            logger.add_image(subtag, x, step)

    @staticmethod
    def log_weight_histogram(module: pl.LightningModule):
        for layer_name, layer in module.named_children():
            for param_name, param in layer.named_parameters():
                tag = f"{layer_name}.{param_name}"
                self.logger.experiment.add_histogram(tag, param, global_step=self.trainer.global_step)


    @staticmethod
    def log_metrics(
        logger: SummaryWriter, 
        prefix: str, 
        global_metrics: dict, 
        local_metrics: Optional[dict] = None, 
        class_names: Optional[dict] = None,
        step: int = 0
    ) -> Dict[str, Tensor]:
        # iterate over classes, logging per-class metrics
        result = {}
        for class_idx in global_metrics.keys():
            class_name = f"class_{class_idx}" if class_names is None else class_names[class_idx]
            super_tag = f"{prefix}{class_name}"

            global_cls = global_metrics[class_idx]
            local_cls = local_metrics[class_idx] if local_metrics is not None else None

            # log global pr curves
            if "type_pairs" in global_cls.keys():
                logger.add_pr_curve(
                    f"{super_tag}/global/type_pr",
                    global_cls["type_pairs"][..., 1], 
                    global_cls["type_pairs"][..., 0],
                    step
                )
            if "malig_pairs" in global_cls.keys():
                logger.add_pr_curve(
                    f"{super_tag}/global/malig_pr",
                    global_cls["malig_pairs"][..., 1], 
                    global_cls["malig_pairs"][..., 0],
                    step
            )

            # log local pr curves
            # TODO malig pairs
            if local_cls is not None and "type_pairs" in local_cls.keys():
                logger.add_pr_curve(
                    f"{super_tag}/local/type_pr",
                    local_cls["type_pairs"][..., 1], 
                    local_cls["type_pairs"][..., 0],
                    step
                )

            # we dont log the rest here, just make a dict for EvalResult
            for k, v in global_cls.items():
                if v.numel() == 1:
                    tag = f"{super_tag}/global/{k}"
                    result[tag] = v

            if local_cls is not None:
                for k, v in local_cls.items():
                    if v.numel() == 1:
                        tag = f"{super_tag}/local/{k}"
                        result[tag] = v

        return result
