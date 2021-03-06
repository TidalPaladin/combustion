#!/usr/bin/env python
# -*- coding: utf-8 -*-


import inspect
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import Callback
from torch import Tensor

from combustion.util import alpha_blend, apply_colormap
from combustion.vision import to_8bit, visualize_bbox


Colormap = Union[str, List[Optional[str]]]
LogFunction = Callable[[Dict[str, Tensor], pl.Trainer, pl.LightningModule, int], None]
ChannelSplits = Union[int, List[int]]
KeypointFunction = Callable[[Tensor, Tensor], Tensor]


def tensorboard_log(targets: Dict[str, Tensor], trainer: pl.Trainer, pl_module: pl.LightningModule, step: int) -> None:
    experiment = pl_module.logger.experiment
    for name, img in targets.items():
        if img.ndim == 4:
            experiment.add_images(name, img, step)
        else:
            experiment.add_image(name, img, step)


def bbox_overlay(img: Tensor, keypoint_dict: Dict[str, Tensor], **kwargs) -> Tensor:
    coords = keypoint_dict["coords"]
    cls = keypoint_dict.get("class", None)
    score = keypoint_dict.get("score", None)

    result = visualize_bbox(img, coords, cls, score, thickness=1, **kwargs)
    return result


class VisualizeCallback(Callback):
    r"""Callback for visualizing image tensors using a PyTorch Lightning logger.

    Example:
        >>> callback = VisualizeCallback("inputs")
        >>>
        >>> # LightningModule.training_step
        >>> def training_step(self, batch, batch_idx):
        >>>     image, target = batch
        >>>     ...
        >>>     # attribute will be logged to TensorBoardLogger under 'train/inputs'
        >>>     self.last_image = image

    Args:
        name (str or list of str):
            Name to be assigned to the logged visualization. When ``split_channels`` is
            selected, ``name`` should be a list of strings assigning names to each split
            section.

        on (str or list of str):
            Modes for which the callback should run. Must be one of
            ``"train"``, ``"val"``, ``"test"``.

        attr_name (str):
            Name of the attribute where the callback will search for the image to be logged.

        epoch_counter (bool):
            If ``True``, report the epoch for each callback invocation. By default, the
            global step is reported.

        max_resolution (tuple of ints, optional):
            If given, resize images to the given :math:`(H, W)` before logging.

        image_limit (int, optional):
            If given, do not log more than ``image_limit`` batches per epoch.

        split_channels (int or list of ints, optional):
            If given, decompose the input tensor using :func:`torch.split`. Each section
            will be assigned the corresponding name in ``name``.

        split_batches (bool):
            If ``True``, log each batched image separately rathern than as part of a shared image.

        interval (int, optional):
            If given, only execute the callback every ``interval`` steps

        resize_mode (str):
            Mode for :func:`torch.nn.functional.interpolate` when ``max_resolution`` is given.

        colormap (str or list of str, optional):
            Colormap to be applied to greyscale images using :func:`combustion.util.apply_colormap`
            When ``split_channels`` is selected, ``colormap`` should be a list of strings specifying
            the color map for each split section.

        ignore_errors (bool):
            If ``True``, do not raise an exception if ``attr_name`` cannot be found.

        log_fn (callable):
            Callable that logs the processed image(s).

        as_uint8 (bool):
            Sets whether or not the processed images will be converted to normalized byte
            tensors before logging.

        per_img_norm (bool):
            Determines if outputs are min-max normalized on a per-image or per-batch basis when
            converting to uint8. By default, ``per_img_norm`` is ``True`` when ``split_batches``
            is ``True``.
    """

    def __init__(
        self,
        name: Union[str, List[str]],
        on: Union[str, Iterable[str]] = ("train", "val", "test"),
        attr_name: str = "last_image",
        epoch_counter: bool = False,
        max_resolution: Optional[Tuple[int, int]] = None,
        image_limit: Optional[int] = None,
        split_channels: Optional[ChannelSplits] = None,
        split_batches: bool = False,
        interval: Optional[int] = None,
        resize_mode: str = "bilinear",
        colormap: Optional[Colormap] = None,
        ignore_errors: bool = False,
        log_fn: LogFunction = tensorboard_log,
        as_uint8: bool = True,
        per_img_norm: Optional[bool] = None,
    ):
        self.name = name
        self.on = tuple(str(x) for x in on) if isinstance(on, Iterable) else (str(on),)
        self.max_resolution = tuple(int(x) for x in max_resolution) if max_resolution is not None else None
        self.split_channels = split_channels
        self.split_batches = bool(split_batches)
        self.resize_mode = str(resize_mode)
        self.attr_name = str(attr_name)
        self.image_limit = int(image_limit) if image_limit is not None else None
        self.colormap = colormap if colormap is not None else None
        self.ignore_errors = bool(ignore_errors)
        self.interval = int(interval) if interval is not None else None
        self.epoch_counter = bool(epoch_counter)
        self.log_fn = log_fn
        self.as_uint8 = bool(as_uint8)
        self.counter = 0
        self.per_img_norm = per_img_norm if per_img_norm is not None else self.split_batches

        if self.split_channels is not None:
            if isinstance(self.name, str):
                raise TypeError(
                    "Expected iterable for `name` when `split_channels` is provided, " f"found {type(self.name)}"
                )
            if not isinstance(self.colormap, (Iterable, type(None))):
                raise TypeError(
                    "Expected iterable for `colormap` when `split_channels` is provided, "
                    f"found {type(self.colormap)}"
                )
        else:
            if not isinstance(self.name, str):
                raise TypeError(
                    "Expected str for `name` when `split_channels` is not provided, " f"found {type(self.name)}"
                )

    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args) -> None:
        with torch.no_grad():
            self._on_batch_end("train", trainer, pl_module)

    def on_validation_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args) -> None:
        with torch.no_grad():
            self._on_batch_end("val", trainer, pl_module)

    def on_test_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args) -> None:
        with torch.no_grad():
            self._on_batch_end("test", trainer, pl_module)

    def _on_batch_end(self, mode: str, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if mode not in self.on:
            return
        if not hasattr(pl_module, self.attr_name):
            if self.ignore_errors:
                return
            else:
                raise AttributeError(f"Module missing expected attribute {self.attr_name}")

        img = getattr(pl_module, self.attr_name)
        step = pl_module.current_epoch if self.epoch_counter else pl_module.global_step

        # skip if enough images have already been logged
        if self.image_limit is not None and self.counter >= self.image_limit:
            return
        # skip if logging is desired at a non-unit interval
        elif self.interval is not None and step % self.interval != 0:
            return

        # single tensor
        if isinstance(img, Tensor):
            images = self._process_image(
                img, self.colormap, self.split_channels, self.name, self.max_resolution, self.as_uint8
            )
            self._log(mode, images, trainer, pl_module, step)
        elif not self.ignore_errors:
            raise TypeError(f"Expected {self.attr_name} to be a tensor or tuple of tensors, " f"but found {type(img)}")

        self.counter += 1

    def on_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        delattr(pl_module, self.attr_name)
        self.counter = 0

    def _process_image(
        self,
        img: Tensor,
        colormap: Optional[Union[str, List[str]]],
        split_channels: Optional[ChannelSplits],
        name: Union[str, List[str]],
        max_resolution: Optional[Tuple[int, int]],
        as_uint8: bool = True,
    ) -> Tensor:
        # split channels
        if split_channels is not None:
            images = torch.split(img, split_channels, dim=1)
            assert isinstance(name, list)
            assert len(name) == len(images)
            images = {n: x for n, x in zip(name, images)}
        else:
            images = {name: img}

        # apply resolution limit
        if max_resolution is not None:
            images = {k: self._apply_log_resolution(v, max_resolution, self.resize_mode) for k, v in images.items()}

        # apply colormap
        if colormap is not None:
            if isinstance(colormap, str):
                colormap = [colormap] * len(images)
            assert len(colormap) == len(images)
            images = {
                k: self.apply_colormap(v, cmap) if cmap is not None else v
                for cmap, (k, v) in zip(colormap, images.items())
            }

        # convert to byte
        if as_uint8:
            images = {k: to_8bit(v, same_on_batch=not self.per_img_norm) for k, v in images.items()}

        return images

    def _log(
        self, mode: str, targets: Dict[str, Tensor], trainer: pl.Trainer, pl_module: pl.LightningModule, step: int
    ) -> None:
        # split batches if requested
        if self.split_batches:
            result = {}
            for name, img in targets.items():
                if img.ndim < 4:
                    continue
                img = torch.split(img, 1, dim=0)
                for idx, img_i in enumerate(img):
                    result[f"{mode}/{name}/{self.counter + idx}"] = img_i
            targets = result
        else:
            targets = {f"{mode}/{n}": v for n, v in targets.items()}

        self.log_fn(targets, trainer, pl_module, step)

    def _apply_log_resolution(self, img: Tensor, limit: Tuple[int, int], mode: str) -> Tensor:
        H, W = img.shape[-2:]
        H_max, W_max = limit

        # noop
        if H > H_max or W > W_max:
            # resize image and boxes if given
            height_ratio, width_ratio = H / H_max, W / W_max
            scale_factor = 1 / max(height_ratio, width_ratio)
            img = F.interpolate(img, scale_factor=scale_factor, mode=mode)

        return img

    @staticmethod
    def apply_colormap(img: Tensor, cmap: str = "gnuplot") -> Tensor:
        # ensure batch dim present
        if img.ndim == 3:
            img = img[None]

        img = apply_colormap(img, cmap=cmap)[:, :3, :, :]
        img = img.squeeze_()
        return img

    @staticmethod
    def alpha_blend(dest: Tensor, src: Tensor, resize_mode="bilinear", *args, **kwargs) -> Tensor:
        # ensure batch dim present
        if dest.ndim == 3:
            dest = dest[None]
        if src.ndim == 3:
            dest = dest[None]

        B1, C1, H1, W1 = dest.shape
        B2, C2, H2, W2 = src.shape

        if C1 != C2:
            if C1 == 1:
                dest = dest.repeat(1, C2, 1, 1)
            elif C2 == 1:
                src = src.repeat(1, C1, 1, 1)
            else:
                raise ValueError(f"could not match shapes {dest.shape}, {src.shape}")

        if (H1, W1) != (H2, W2):
            src = F.interpolate(src, (H1, W1), mode=resize_mode)

        blended, _ = alpha_blend(src, dest, *args, **kwargs)
        return blended.view_as(dest)

    def __repr__(self):
        s = f"{self.__class__.__name__}(name='{self.name}'"
        init_signature = inspect.signature(self.__class__)
        defaults = {
            k: v.default for k, v in init_signature.parameters.items() if v.default is not inspect.Parameter.empty
        }

        for name, default in defaults.items():
            val = getattr(self, name)
            display_val = f"'{val}'" if isinstance(val, str) else val
            if val != default:
                s += f", {name}={display_val}"
        s += ")"
        return s


class KeypointVisualizeCallback(VisualizeCallback):
    r"""Callback for visualizing image tensors and associated keypoint / anchor boxes
    using a PyTorch Lightning logger. The calling model should assign an attribute
    with a tuple of (image, keypoint information).

    Example:
        >>> callback = VisualizeCallback("inputs")
        >>>
        >>> # LightningModule.training_step
        >>> def training_step(self, batch, batch_idx):
        >>>     image, (box_coords, box_classes) = batch
        >>>     ...
        >>>     # attribute will be logged to TensorBoardLogger under 'train/inputs'
        >>>     self.last_image = image, {"coords": box_coords, "class": box_classes}

    Args:
        name (str or list of str):
            Name to be assigned to the logged visualization. When ``split_channels`` is
            selected, ``name`` should be a list of strings assigning names to each split
            section.

        on (str or list of str):
            Modes for which the callback should run. Must be one of
            ``"train"``, ``"val"``, ``"test"``.

        attr_name (str):
            Name of the attribute where the callback will search for the image to be logged.

        epoch_counter (bool):
            If ``True``, report the epoch for each callback invocation. By default, the
            global step is reported.

        max_resolution (tuple of ints, optional):
            If given, resize images to the given :math:`(H, W)` before logging.

        image_limit (int, optional):
            If given, do not log more than ``image_limit`` batches per epoch.

        split_channels (int or list of ints, optional):
            If given, decompose the input tensor using :func:`torch.split`. Each section
            will be assigned the corresponding name in ``name``.

        split_batches (bool):
            If ``True``, log each batched image separately rathern than as part of a shared image.

        interval (int, optional):
            If given, only execute the callback every ``interval`` steps

        resize_mode (str):
            Mode for :func:`torch.nn.functional.interpolate` when ``max_resolution`` is given.

        colormap (str or list of str, optional):
            Colormap to be applied to greyscale images using :func:`combustion.util.apply_colormap`
            When ``split_channels`` is selected, ``colormap`` should be a list of strings specifying
            the color map for each split section.

        ignore_errors (bool):
            If ``True``, do not raise an exception if ``attr_name`` cannot be found.

        log_fn (callable):
            Callable that logs the processed image(s). By default, :func:`VisualizeCallback.tensorboard_log`
            is used to provide logging suitable for :class:`pytorch_lightning.loggers.TensorBoardLogger`.
        as_uint8 (bool):
            Sets whether or not the processed images will be converted to normalized byte
            tensors before logging.

        per_img_norm (bool):
            Determines if outputs are min-max normalized on a per-image or per-batch basis when
            converting to uint8. By default, ``per_img_norm`` is ``True`` when ``split_batches``
            is ``True``.

        overlay_keypoints (callable):
            Function that accepts an image and a dictionary of keypoint tensors for that image, and
            produces an output image tensor with keypoint information added to the image.
    """

    def __init__(
        self,
        name: Union[str, List[str]],
        on: Union[str, List[str]] = ["train", "val", "test"],
        attr_name: str = "last_image",
        epoch_counter: bool = False,
        max_resolution: Optional[Tuple[int, int]] = None,
        image_limit: Optional[int] = None,
        split_channels: Optional[ChannelSplits] = None,
        split_batches: bool = False,
        interval: Optional[int] = None,
        resize_mode: str = "bilinear",
        colormap: Optional[Colormap] = None,
        ignore_errors: bool = False,
        log_fn: LogFunction = tensorboard_log,
        as_uint8: bool = True,
        per_img_norm: Optional[bool] = None,
        overlay_keypoints: KeypointFunction = bbox_overlay,
    ):
        super().__init__(
            name,
            on,
            attr_name,
            epoch_counter,
            max_resolution,
            image_limit,
            split_channels,
            split_batches,
            interval,
            resize_mode,
            colormap,
            ignore_errors,
            log_fn,
            as_uint8,
            per_img_norm,
        )
        if overlay_keypoints is not None:
            self.overlay_keypoints = overlay_keypoints
        else:
            self.overlay_keypoints = KeypointVisualizeCallback.bbox_overlay

    def _on_batch_end(self, mode: str, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not hasattr(pl_module, self.attr_name):
            if self.ignore_errors:
                return
            else:
                raise AttributeError(f"Module missing expected attribute {self.attr_name}")

        img, keypoint_dict = getattr(pl_module, self.attr_name)
        step = pl_module.current_epoch if self.epoch_counter else pl_module.global_step

        # skip if enough images have already been logged
        if self.image_limit is not None and self.counter >= self.image_limit:
            return
        # skip if logging is desired at a non-unit interval
        elif self.interval is not None and step % self.interval != 0:
            return

        if isinstance(img, Tensor):
            # this error can't be handled w/ ignore_errors
            if "coords" not in keypoint_dict.keys():
                raise KeyError("expected coords key in keypoint dict")

            images = self._process_image(
                img, keypoint_dict, self.colormap, self.split_channels, self.name, self.max_resolution, self.as_uint8
            )
            self._log(mode, images, trainer, pl_module, step)
        elif not self.ignore_errors:
            raise TypeError(f"Expected {self.attr_name} to be a tensor or tuple of tensors, " f"but found {type(img)}")
        self.counter += 1

    def _process_image(
        self,
        img: Tensor,
        keypoint_dict: Dict[str, Tensor],
        colormap: Optional[Union[str, List[str]]],
        split_channels: Optional[ChannelSplits],
        name: Union[str, List[str]],
        max_resolution: Optional[Tuple[int, int]],
        as_uint8: bool = True,
    ) -> Tensor:

        B, C, H, W = img.shape
        images = super()._process_image(img, colormap, split_channels, name, max_resolution, False)

        # apply resolution limit to keypoints
        if max_resolution is not None:
            H_scaled, W_scaled = next(iter(images.values())).shape[-2:]
            if H_scaled / H != 1.0:
                coords = keypoint_dict["coords"].clone().float()
                padding = coords < 0
                coords.mul_(H_scaled / H)
                coords[padding] = -1
                keypoint_dict["coords"] = coords

        # overlay keypoints, accounting for channel splitting
        if self.split_channels:
            keypoints = [keypoint_dict] * C
            assert len(keypoints) == C
            images = {k: self.overlay_keypoints(v, d) for d, (k, v) in zip(keypoints, images.items())}
        else:
            images = {k: self.overlay_keypoints(v, keypoint_dict) for k, v in images.items()}

        if as_uint8:
            images = {k: to_8bit(v, same_on_batch=not self.per_img_norm) for k, v in images.items()}

        return images


class BlendVisualizeCallback(VisualizeCallback):
    r"""Callback for visualizing image tensors and associated keypoint / anchor boxes
    using a PyTorch Lightning logger. The calling model should assign an attribute
    with a tuple of (image, keypoint information).

    Example:
        >>> callback = VisualizeCallback("inputs", colormap=(None, "gnuplot"))
        >>>
        >>> # LightningModule.training_step
        >>> def training_step(self, batch, batch_idx):
        >>>     image, target = batch
        >>>     ...
        >>>     # target will be converted to RGB w/ gnuplot colormap and alpha blended with image
        >>>     self.last_image = image, target

    Args:
        name (str or list of str):
            Name to be assigned to the logged visualization. When ``split_channels`` is
            selected, ``name`` should be a list of strings assigning names to each split
            section.

        on (str or list of str):
            Modes for which the callback should run. Must be one of
            ``"train"``, ``"val"``, ``"test"``.

        attr_name (str):
            Name of the attribute where the callback will search for the image to be logged.

        epoch_counter (bool):
            If ``True``, report the epoch for each callback invocation. By default, the
            global step is reported.

        max_resolution (tuple of ints, optional):
            If given, resize images to the given :math:`(H, W)` before logging.

        image_limit (int, optional):
            If given, do not log more than ``image_limit`` batches per epoch.

        split_channels (int or list of ints, optional):
            If given, decompose the input tensor using :func:`torch.split`. Each section
            will be assigned the corresponding name in ``name``.

        split_batches (bool):
            If ``True``, log each batched image separately rathern than as part of a shared image.

        interval (int, optional):
            If given, only execute the callback every ``interval`` steps

        resize_mode (str):
            Mode for :func:`torch.nn.functional.interpolate` when ``max_resolution`` is given.

        colormap (str or list of str, optional):
            Colormap to be applied to greyscale images using :func:`combustion.util.apply_colormap`
            When ``split_channels`` is selected, ``colormap`` should be a list of strings specifying
            the color map for each split section.

        ignore_errors (bool):
            If ``True``, do not raise an exception if ``attr_name`` cannot be found.

        log_fn (callable):
            Callable that logs the processed image(s). By default, :func:`VisualizeCallback.tensorboard_log`
            is used to provide logging suitable for :class:`pytorch_lightning.loggers.TensorBoardLogger`.
        as_uint8 (bool):
            Sets whether or not the processed images will be converted to normalized byte
            tensors before logging.

        per_img_norm (bool):
            Determines if outputs are min-max normalized on a per-image or per-batch basis when
            converting to uint8. By default, ``per_img_norm`` is ``True`` when ``split_batches``
            is ``True``.

        blend_func (callable):
            TODO
    """

    def __init__(
        self,
        name: Union[str, List[str]],
        on: Union[str, List[str]] = ["train", "val", "test"],
        attr_name: str = "last_image",
        epoch_counter: bool = False,
        max_resolution: Optional[Tuple[int, int]] = None,
        image_limit: Optional[int] = None,
        split_channels: Tuple[Optional[ChannelSplits], Optional[ChannelSplits]] = (None, None),
        split_batches: bool = False,
        interval: Optional[int] = None,
        resize_mode: str = "bilinear",
        colormap: Tuple[Optional[Colormap], Optional[Colormap]] = (None, None),
        ignore_errors: bool = False,
        log_fn: LogFunction = tensorboard_log,
        as_uint8: bool = True,
        per_img_norm: Optional[bool] = None,
        alpha: Tuple[float, float] = (1.0, 0.5),
    ):
        if isinstance(split_channels, int) or (isinstance(split_channels, Iterable) and len(split_channels) != 2):
            split_channels = (split_channels, None)
        if isinstance(colormap, str) or (isinstance(colormap, Iterable) and len(colormap) != 2):
            colormap = (colormap, None)

        super().__init__(
            name,
            on,
            attr_name,
            epoch_counter,
            max_resolution,
            image_limit,
            split_channels[0],
            split_batches,
            interval,
            resize_mode,
            colormap[0],
            ignore_errors,
            log_fn,
            as_uint8,
            per_img_norm,
        )
        self.split_channels = tuple(split_channels)
        self.colormap = tuple(colormap)
        self.alpha = alpha

    def _on_batch_end(self, mode: str, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not hasattr(pl_module, self.attr_name):
            if self.ignore_errors:
                return
            else:
                raise AttributeError(f"Module missing expected attribute {self.attr_name}")

        dest, src = getattr(pl_module, self.attr_name)
        step = pl_module.current_epoch if self.epoch_counter else pl_module.global_step

        # skip if enough images have already been logged
        if self.image_limit is not None and self.counter >= self.image_limit:
            return
        # skip if logging is desired at a non-unit interval
        elif self.interval is not None and step % self.interval != 0:
            return

        if isinstance(dest, Tensor) and isinstance(src, Tensor):
            images = self._process_image(
                dest, src, self.colormap, self.split_channels, self.name, self.max_resolution, self.as_uint8
            )
            self._log(mode, images, trainer, pl_module, step)
        elif not self.ignore_errors:
            raise TypeError(
                f"Expected {self.attr_name} to be a tuple of tensors, " f"but found {type(dest)}, {type(src)}"
            )
        self.counter += 1

    def _process_image(
        self,
        dest: Tensor,
        src: Tensor,
        colormap: Tuple[Optional[Colormap], Optional[Colormap]],
        split_channels: Tuple[Optional[ChannelSplits], Optional[ChannelSplits]],
        name: Union[str, List[str]],
        max_resolution: Optional[Tuple[int, int]],
        as_uint8: bool = True,
    ) -> Tensor:

        B, C, H, W = dest.shape
        if any(split_channels):
            name1, name2 = (name if split is not None else "__name__" for split in split_channels)
        else:
            name1, name2 = name, name

        images_dest = super()._process_image(dest, colormap[0], split_channels[0], name1, max_resolution, False)
        images_src = super()._process_image(src, colormap[1], split_channels[1], name2, max_resolution, False)

        if len(images_dest) != len(images_src):
            if len(images_dest) == 1:
                val = next(iter(images_dest.values()))
                images_dest = {k: val for k in images_src.keys()}
            elif len(images_src) == 1:
                val = next(iter(images_src.values()))
                images_src = {k: val for k in images_dest.keys()}
            else:
                raise ValueError(
                    f"Unable to broadcast processed images:\n"
                    f"Destination dict:\n {k: v.shape for k, v in images_dest.items()}\n\n"
                    f"Source dict:\n {k: v.shape for k, v in images_src.items()}"
                )

        dest_alpha, src_alpha = self.alpha
        images = {
            k1
            if "__name__" not in k1
            else k2: VisualizeCallback.alpha_blend(d, s, self.resize_mode, dest_alpha=dest_alpha, src_alpha=src_alpha)
            for (k1, d), (k2, s) in zip(images_dest.items(), images_src.items())
        }
        assert "__name__" not in images.keys()

        if as_uint8:
            images = {k: to_8bit(v, same_on_batch=not self.per_img_norm) for k, v in images.items()}

        return images
