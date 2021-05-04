#!/usr/bin/env python
# -*- coding: utf-8 -*-


from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from PIL import Image
from pytorch_lightning.callbacks import Callback
from torch import Tensor
from torchvision.utils import make_grid

from combustion.util import alpha_blend, apply_colormap
from combustion.vision import to_8bit, visualize_bbox

from .base import AttributeCallback, mkdir, resolve_dir


Colormap = Union[str, List[Optional[str]]]
LogFunction = Callable[[Dict[str, Tensor], pl.Trainer, pl.LightningModule, int, Optional[int], AttributeCallback], None]
ChannelSplits = Union[int, List[int]]
KeypointFunction = Callable[[Tensor, Tensor], Tensor]


def tensorboard_log(
    targets: Dict[str, Tensor],
    trainer: pl.Trainer,
    pl_module: pl.LightningModule,
    step: int,
    batch_idx: Optional[int],
    caller: Callback,
) -> None:
    experiment = pl_module.logger.experiment
    for name, img in targets.items():
        if img.ndim == 4:
            experiment.add_images(name, img, step)
        else:
            experiment.add_image(name, img, step)


class ImageSave:
    r"""Log function for :class:`VisualizeCallback` that saves image tensors as PNG files.

    Args:
        path (:class:`Path`):
            Path where images will be saved. Defaults to ``trainer.default_root_dir``.

        quality (int):
            Quality value 1-100 for :func:`PIL.Image.save`

    Example:
        >>> log_fn = ImageSave()
        >>> callback = VisualizeCallback("inputs", log_fn=log_fn)
        >>>
        >>> # LightningModule.training_step
        >>> def training_step(self, batch, batch_idx):
        >>>     image, target = batch
        >>>     ...
        >>>     # attribute will be logged to trainer.default_root_dir under 'train/inputs_i/batch_i.png'
        >>>     self.last_image = image
    """

    def __init__(self, path: Optional[Path] = None, quality: int = 95):
        self._path = Path(path) if path is not None else None
        self.path = None
        self.quality = int(quality)

    @staticmethod
    def save_image(data: Tensor, dest: Path, quality: int = 95) -> None:
        if data.ndim > 3 or data.ndim < 2:
            raise ValueError(f"Invalid tensor of shape {data.shape}")
        elif data.ndim == 3:
            data = data.permute(1, 2, 0)  # channels last
        assert data.ndim == 2 or data.shape[-1] in (1, 3, 4)
        img = Image.fromarray(data.cpu().numpy())
        dest.parent.mkdir(exist_ok=True, parents=True)
        img.save(str(dest), quality=quality)

    @staticmethod
    def save_batch(data: Tensor, dest: Path, quality: int = 95) -> None:
        if data.ndim < 4:
            raise ValueError(f"Invalid tensor of shape {data.shape}")
        grid = make_grid(data)
        ImageSave.save_image(grid, dest, quality)

    def __call__(
        self,
        targets: Dict[str, Tensor],
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        step: int,
        batch_idx: Optional[int],
        caller: Callback,
    ) -> None:
        if self.path is None:
            self.path = Path(resolve_dir(trainer, self._path, "saved_images"))

        root = Path(self.path)
        for name, img in targets.items():
            dest = Path(root, name, caller.read_step_as_str(pl_module, batch_idx)).with_suffix(".png")
            mkdir(dest.parent, trainer)
            if img.ndim == 4:
                self.save_batch(img, dest, self.quality)
            else:
                self.save_image(img, dest, self.quality)


def bbox_overlay(img: Tensor, keypoint_dict: Dict[str, Tensor], **kwargs) -> Tensor:
    coords = keypoint_dict["coords"]
    cls = keypoint_dict.get("class", None)
    score = keypoint_dict.get("score", None)
    names = keypoint_dict.get("names", None)

    result = visualize_bbox(img, coords, cls, score, names, thickness=1, **kwargs)
    return result


class VisualizeCallback(AttributeCallback):
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

        triggers (str or list of str):
            Modes for which the callback should run. Must be one of
            ``"train"``, ``"val"``, ``"test"``.

        hook (str):
            One of ``"step"`` or ``"epoch"``, determining if the callback will be triggered on
            batch end or epoch end.

        attr_name (str):
            Name of the attribute where the callback will search for the image to be logged.

        epoch_counter (bool):
            If ``True``, report the epoch for each callback invocation. By default, the
            global step is reported.

        max_resolution (tuple of ints, optional):
            If given, resize images to the given :math:`(H, W)` before logging.

        max_calls (int, optional):
            If given, do not log more than ``max_calls`` batches per epoch.

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

        log_fn (callable or iterable of callables):
            Callable(s) that logs the processed image(s).

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
        triggers: Union[str, Iterable[str]] = ("train", "val", "test"),
        hook: str = "step",
        attr_name: str = "last_image",
        epoch_counter: bool = False,
        max_resolution: Optional[Tuple[int, int]] = None,
        max_calls: Optional[int] = None,
        split_channels: Optional[ChannelSplits] = None,
        split_batches: bool = False,
        interval: Optional[int] = None,
        resize_mode: str = "bilinear",
        colormap: Optional[Colormap] = None,
        ignore_errors: bool = False,
        log_fn: Union[LogFunction, Iterable[LogFunction]] = tensorboard_log,
        as_uint8: bool = True,
        per_img_norm: Optional[bool] = None,
    ):
        super().__init__(
            triggers,
            hook,
            attr_name,
            epoch_counter,
            max_calls,
            interval,
            ignore_errors,
        )

        self.name = name
        self.max_resolution = tuple(int(x) for x in max_resolution) if max_resolution is not None else None
        self.split_channels = split_channels
        self.split_batches = bool(split_batches)
        self.resize_mode = str(resize_mode)
        self.colormap = colormap if colormap is not None else None
        self.log_fn = log_fn
        self.as_uint8 = bool(as_uint8)
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

    def callback_fn(
        self,
        hook: Tuple[str, str],
        attr: Any,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        step: int,
        batch_idx: Optional[int],
    ) -> None:
        # single tensor
        _hook, _mode = hook
        if isinstance(attr, Tensor):
            images = self._process_image(
                attr, self.colormap, self.split_channels, self.name, self.max_resolution, self.as_uint8
            )
            self._log(_mode, images, trainer, pl_module, step, batch_idx)
        elif not self.ignore_errors:
            raise TypeError(f"Expected {self.attr_name} to be a tensor or tuple of tensors, " f"but found {type(attr)}")

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
        self,
        mode: str,
        targets: Dict[str, Tensor],
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        step: int,
        batch_idx: Optional[int],
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

        if isinstance(self.log_fn, Iterable):
            for f in self.log_fn:
                f(targets, trainer, pl_module, step, batch_idx, self)
        else:
            self.log_fn(targets, trainer, pl_module, step, batch_idx, self)

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
        elif dest.ndim > 4:
            dest = dest.view(-1, *dest.shape[-3:])
        if src.ndim == 3:
            src = src[None]
        elif dest.ndim > 4:
            src = src.view(-1, *src.shape[-3:])

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
            src = F.interpolate(src, (H1, W1), mode=resize_mode, align_corners=True)

        blended, _ = alpha_blend(src, dest, *args, **kwargs)
        return blended.view_as(dest)


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

        triggers (str or list of str):
            Modes for which the callback should run. Must be one of
            ``"train"``, ``"val"``, ``"test"``.

        hook (str):
            One of ``"step"`` or ``"epoch"``, determining if the callback will be triggered on
            batch end or epoch end.

        attr_name (str):
            Name of the attribute where the callback will search for the image to be logged.

        epoch_counter (bool):
            If ``True``, report the epoch for each callback invocation. By default, the
            global step is reported.

        max_resolution (tuple of ints, optional):
            If given, resize images to the given :math:`(H, W)` before logging.

        max_calls (int, optional):
            If given, do not log more than ``max_calls`` batches per epoch.

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

        log_fn (callable or iterable of callables):
            Callable(s) that logs the processed image(s). By default, :func:`VisualizeCallback.tensorboard_log`
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
        triggers: Union[str, List[str]] = ["train", "val", "test"],
        hook: str = "step",
        attr_name: str = "last_image",
        epoch_counter: bool = False,
        max_resolution: Optional[Tuple[int, int]] = None,
        max_calls: Optional[int] = None,
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
            triggers,
            hook,
            attr_name,
            epoch_counter,
            max_resolution,
            max_calls,
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

    def callback_fn(
        self,
        hook: Tuple[str, str],
        attr: Any,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        step: int,
        batch_idx: Optional[int],
    ) -> None:
        _hook, _mode = hook
        if not hasattr(attr, "__len__") or len(attr) != 2:
            if self.ignore_errors:
                return
            else:
                raise TypeError(f"Expected {self.attr_name} to be a 2-tuple of tensor and dict, " f"but found {attr}")

        img, keypoint_dict = attr
        img: Tensor
        keypoint_dict: Dict[str, Tensor]

        if isinstance(img, Tensor):
            # this error can't be handled w/ ignore_errors
            if "coords" not in keypoint_dict.keys():
                raise KeyError("expected coords key in keypoint dict")

            images = self._process_image(
                img, keypoint_dict, self.colormap, self.split_channels, self.name, self.max_resolution, self.as_uint8
            )
            self._log(_mode, images, trainer, pl_module, step, batch_idx)
        elif not self.ignore_errors:
            raise TypeError(
                f"Expected {self.attr_name}[0] to be a tensor or tuple of tensors, " f"but found {type(img)}"
            )

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

        triggers (str or list of str):
            Modes for which the callback should run. Must be one of
            ``"train"``, ``"val"``, ``"test"``.

        hook (str):
            One of ``"step"`` or ``"epoch"``, determining if the callback will be triggered on
            batch end or epoch end.

        attr_name (str):
            Name of the attribute where the callback will search for the image to be logged.

        epoch_counter (bool):
            If ``True``, report the epoch for each callback invocation. By default, the
            global step is reported.

        max_resolution (tuple of ints, optional):
            If given, resize images to the given :math:`(H, W)` before logging.

        max_calls (int, optional):
            If given, do not log more than ``max_calls`` batches per epoch.

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

        log_fn (callable or iterable of callables):
            Callable(s) that logs the processed image(s). By default, :func:`VisualizeCallback.tensorboard_log`
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
        triggers: Union[str, List[str]] = ["train", "val", "test"],
        hook: str = "step",
        attr_name: str = "last_image",
        epoch_counter: bool = False,
        max_resolution: Optional[Tuple[int, int]] = None,
        max_calls: Optional[int] = None,
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
            triggers,
            hook,
            attr_name,
            epoch_counter,
            max_resolution,
            max_calls,
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

    def callback_fn(
        self,
        hook: Tuple[str, str],
        attr: Any,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        step: int,
        batch_idx: Optional[int],
    ) -> None:
        _hook, _mode = hook
        if not hasattr(attr, "__len__") or len(attr) != 2:
            if self.ignore_errors:
                return
            else:
                raise TypeError(f"Expected {self.attr_name} to be a 2-tuple of tensor and dict, " f"but found {attr}")

        dest, src = attr
        if isinstance(dest, Tensor) and isinstance(src, Tensor):
            images = self._process_image(
                dest, src, self.colormap, self.split_channels, self.name, self.max_resolution, self.as_uint8
            )
            self._log(_mode, images, trainer, pl_module, step, batch_idx)
        elif not self.ignore_errors:
            raise TypeError(
                f"Expected {self.attr_name} to be a tuple of tensors, " f"but found {type(dest)}, {type(src)}"
            )

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
