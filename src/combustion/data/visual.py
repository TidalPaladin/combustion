#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import os
from typing import Tuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ..util.pytorch import check_ndim_match
from .loader import load_data


def check_data(args):
    ds = load_data(args, "train")
    for batch in ds:
        for i, (frames, labels) in enumerate(batch):
            label = labels.squeeze(0).squeeze(0).detach().numpy()
            frame = frames[0, frames.shape[-3] // 2, ...].detach().numpy()
            filename = "input_%d.png" % i
            path = os.path.join(args.result_path, filename)
            visualize_image(frame, label, path)
        break


def visualize_image(
    data: np.array = None,
    label: np.array = None,
    true_label: np.array = None,
    filename: str = None,
    title: str = None,
    interpolation: Tuple[str, str] = ("none", "none"),
    cmap: Tuple[str, str] = ("gray", "gnuplot"),
    alpha: float = 0.5,
    size=(8, 6),
    dpi=80,
    **kwargs,
) -> Figure:
    """visualize_image

    :param data: the image to visualize 
    :param label: label image overlay
    :param filename: filename to save image as
    :param title: title for the figure
    :param interpolation: pyplot interpolation strategy
    :param cmap: tuple of pyplot color maps for data and label respectively
    :param alpha: alpha to use for label image overlay
    :param kwargs: forwarded to pyplot.savefig

    :rtype: Figure The generated figure
    """
    img_cmap, label_cmap = cmap
    data_interp, label_interp = interpolation

    if all([x is None for x in (data, label, true_label)]):
        raise ValueError("frames, labels, and")

    if label is not None:
        check_ndim_match(data, label, "data", "labels")
    if true_label is not None:
        check_ndim_match(data, true_label, "data", "true_label")

    if true_label is not None and label is not None:
        if data.ndim != label.ndim:
            raise ValueError(f"data label ndim mismatch: {data.shape} vs {label.shape}")
        if data.ndim != true_label.ndim:
            raise ValueError(f"data true_label ndim mismatch: {data.shape} vs {true_label.shape}")

        fig, axs = plt.subplots(1, 2, figsize=size, dpi=dpi)
        fig.subplots_adjust(left=0.05, bottom=0.06, right=0.95, top=0.94, wspace=0.1)
        fig.suptitle(title)
        fig.tight_layout()
        ax1, ax2 = axs.ravel()
        y, x = true_label.nonzero()

        im1 = ax1.imshow(data, cmap=img_cmap, interpolation=data_interp)
        im2 = ax1.imshow(label, cmap=label_cmap, alpha=alpha, interpolation=label_interp)
        ax1.tick_params(bottom=False, top=False, left=False, right=False)
        ax1.set_title("Predicted Intensities")
        ax1.set_yticklabels([])
        ax1.set_xticklabels([])
        _ = fig.colorbar(im2, ax=ax1, fraction=0.046, pad=0.05, orientation="horizontal")
        _.ax.ticklabel_format(scilimits=(-2, 2))

        error = abs(true_label - label)
        im3 = ax2.imshow(data, cmap=img_cmap, interpolation=data_interp)
        im4 = ax2.imshow(error, cmap=label_cmap, alpha=alpha, interpolation=label_interp)
        ax2.tick_params(bottom=False, top=False, left=False, right=False)
        ax2.set_yticklabels([])
        ax2.set_xticklabels([])
        ax2.set_title("Absolute Error")
        _ = fig.colorbar(im4, ax=ax2, fraction=0.046, pad=0.05, orientation="horizontal")
        _.ax.ticklabel_format(scilimits=(-2, 2))

        txt = ("Total Microbubbles: {}\n" "Model Outputs > 0: {}\n" "Model Outputs >= 0.5: {}").format(
            len((true_label == 1).nonzero()[0]), len(label.nonzero()[0]), len((label > 0.5).nonzero()[0])
        )
        fig.text(0.5, 0.8, txt, ha="center")

    else:
        fig, ax = plt.subplots(figsize=size, dpi=dpi)
        fig.suptitle(title)
        im = ax.imshow(data, cmap=img_cmap, interpolation=data_interp)
        if label is not None:
            ax.imshow(label, cmap=label_cmap, alpha=alpha, interpolation=label_interp)
            ax.set_title("Predicted Intensities")
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, ax=cax)

    if filename is not None:
        logging.debug("Saving figure to %s", filename)
        plt.savefig(filename, dpi=140, **kwargs)
    return fig


def visualize_video(
    data: np.array,
    fps: float,
    labels: np.array = None,
    filename: str = None,
    title: str = None,
    interpolation: str = "none",
    cmap: Tuple[str, str] = ("gray", "gnuplot"),
    alpha: float = 0.5,
    **kwargs,
) -> animation.Animation:
    """visualize_image

    :param data: the image to visualize 
    :param fps: the framerate of the animation
    :param label: label image overlay
    :param filename: filename to save image as
    :param title: title for the figure
    :param interpolation: pyplot interpolation strategy
    :param cmap: tuple of pyplot color maps for data and label respectively
    :param alpha: alpha to use for label image overlay
    :param kwargs: forwarded to pyplot.savefig

    :rtype: Animation The generated pyplot animation
    """
    if labels is not None:
        check_ndim_match(data, labels, "data", "labels")
    img_cmap, label_cmap = cmap

    # create labeled figure and first frame of data
    fig, ax = plt.subplots()
    fig.suptitle(title)
    im = plt.imshow(data[0], cmap=img_cmap, interpolation=interpolation)
    if labels is not None:
        lb = plt.imshow(labels[0], cmap=label_cmap, alpha=alpha, interpolation=interpolation)

    # called by animator to update each frame
    def update(frame):
        f = data[frame]
        im.set_array(f)
        if labels is not None:
            mask = labels[frame]
            lb.set_array(mask)
            return im, lb
        return (im,)

    interval, frames = 1000.0 / fps, len(data)
    ani = animation.FuncAnimation(fig, update, frames=frames, interval=interval, blit=True)

    logging.info("Saving figure to %s", filename)
    if filename is not None:
        logging.info("Saving animation to %s", filename)
        ani.save(filename, **kwargs)
    return ani


def visualize_mb(
    data: np.array = None,
    label: np.array = None,
    true_label: np.array = None,
    filename: str = None,
    title: str = None,
    interpolation: Tuple[str, str] = ("none", "none"),
    cmap: Tuple[str, str] = ("gray", "gnuplot"),
    alpha: float = 0.5,
    size=(8, 6),
    dpi=80,
    mb_size=40,
    limit_mb=5,
    **kwargs,
) -> Figure:
    """visualize_image

    :param data: the image to visualize 
    :param label: label image overlay
    :param filename: filename to save image as
    :param title: title for the figure
    :param interpolation: pyplot interpolation strategy
    :param cmap: tuple of pyplot color maps for data and label respectively
    :param alpha: alpha to use for label image overlay
    :param kwargs: forwarded to pyplot.savefig

    :rtype: Figure The generated figure
    """
    if true_label is None:
        raise ValueError("true label is required")

    # TODO cleanup, can probably avoid torch
    mb_centers = torch.nonzero(torch.from_numpy(true_label))

    for i, coords in enumerate(mb_centers):
        if i > limit_mb:
            break

        h, w = coords

        h_low = max(h - mb_size // 2, 0)
        h_high = min(h + mb_size // 2, true_label.shape[-2])
        w_low = max(w - mb_size // 2, 0)
        w_high = min(w + mb_size // 2, true_label.shape[-1])

        mb_data = data[h_low:h_high, w_low:w_high]
        mb_label = label[h_low:h_high, w_low:w_high]
        mb_true_label = true_label[h_low:h_high, w_low:w_high]

        if filename is not None:
            name, ext = filename.split(".")
            name += f"_bubble_{i}"
            name += ext
        else:
            name = None

        fig = visualize_image(
            mb_data, mb_label, mb_true_label, name, title, interpolation, cmap, alpha, size, dpi, **kwargs,
        )
        yield fig


def _check_shapes_match(data, label):
    if not data.shape == label.shape:
        raise ValueError("data shape {} must match label shape {}".format(data.shape, label.shape))
