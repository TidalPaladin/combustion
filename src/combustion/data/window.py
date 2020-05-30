#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from itertools import islice
from typing import Generator, Iterable, Tuple

import torch
from torch import Tensor


class Window(ABC):
    r"""Helper to apply a window over an iterable or set of indices.

    Args:

        before (int, optional):
            The number of prior elements to include in the window.

        after (int, optional):
            The number of proceeding elements to include in the window.

    """

    def __init__(self, before: int = 0, after: int = 0):
        if int(before) < 0:
            raise ValueError("before must be int >= 0")
        if int(after) < 0:
            raise ValueError("before must be int >= 0")
        self.before = int(before)
        self.after = int(after)
        if not (self.before or self.after):
            raise ValueError("before or after must not be 0")

    def __len__(self):
        """Returns the number of frames in the window."""
        return self.before + self.after + 1

    def __call__(self, examples: Iterable) -> Generator[Tuple[Tensor, Tensor], None, None]:
        r"""Generates frames, labels tuples by applying the window to
        ``examples``. For an input/labels of shape :math:`CxHxW`, and a window of length
        :math:`D`, the output will be of frame, label tuples of shape
        (:math:`CxDxHxW`, :math:`CxHxW`).

        Args:

            examples (Iterable):
                An iterable of (frame, label) tuples to be windowed

        Returns:
            A generator that yields torch.Tensor tuples with the window
            function applied.
        """
        # method to efficiently yield window tuples from iterable
        def raw_window():
            it = iter(examples)
            slices = tuple(islice(it, len(self)))
            if len(slices) == len(self):
                yield slices
            for elem in it:
                slices = slices[1:] + (elem,)
                yield slices

        # parse windowed examples into higher rank tensors
        depth_dim = 0
        midpoint = self.before

        for window in raw_window():
            indices = self.indices(midpoint)
            window = tuple([window[x - indices[0]] for x in indices])
            frames, labels = list(zip(*window))
            frames = [torch.as_tensor(f) for f in frames]
            labels = [torch.as_tensor(l) for l in labels]
            frames = torch.stack(frames, depth_dim).refine_names("D", "H", "W")
            label = labels[midpoint].refine_names("H", "W")
            yield frames.align_to("C", "D", "H", "W"), label.align_to("C", "H", "W")

    def __repr__(self):
        name = self.__class__.__name__
        return f"{name}(before={self.before}, after={self.after})"

    @abstractmethod
    def indices(self, pos: int) -> Tuple[int, ...]:
        r"""Given an index ``pos``, return a tuple of indices that are
        part of the window centered at ``pos``.

        Args:

            pos (int): The index of the window center

        Returns:

            A tuple of ints giving the indices of a window centered at ``pos``.
        """
        low = pos - self.before
        high = pos + self.after
        return tuple(set(range(low, high)))

    def estimate_size(self, num_frames: int) -> int:
        r"""Given a number of examples in the un-windowed input,
        estimate the number of examples in the windowed result.

        Args:
            num_frames (int):
                The number of frames in the un-windowed dataset.

        Returns:
            Estimated number of frames in the windowed output.
        """
        return num_frames - (self.before + self.after)


class DenseWindow(Window):
    r"""Helper to apply a dense window over an iterable or set of indices.
    A dense window includes all indices from ``center-before`` to ``center+after``.
    For a window that includes only frames (``center-before``, ``center``, ``center+after``),
    see SparseWindow.


    Args:

        before (int, optional):
            The number of prior elements to include in the window.

        after (int, optional):
            The number of proceeding elements to include in the window.

    """

    def indices(self, pos: int) -> Tuple[int, ...]:
        r"""Given an index ``pos``, return a tuple of indices that are
        part of the window centered at ``pos``.

        Args:

            pos (int): The index of the window center

        Returns:

            A tuple of ints giving the indices of a window centered at ``pos``.
        """
        low = pos - self.before
        high = pos + self.after
        return tuple(set(range(low, high + 1)))


class SparseWindow(Window):
    r"""Helper to apply a sparse window over an iterable or set of indices.
    A sparse window only includes frames (``center-before``, ``center``, ``center+after``).
    For a window that includes all indices from ``center-before`` to ``center+after``, see
    :class:`DenseWindow`


    Args:

        before (int, optional):
            The number of prior elements to include in the window.

        after (int, optional):
            The number of proceeding elements to include in the window.

    """

    def indices(self, pos: int) -> Tuple[int, ...]:
        r"""Given an index ``pos``, return a tuple of indices that are
        part of the window centered at ``pos``.

        Args:

            pos (int): The index of the window center

        Returns:

            A tuple of ints giving the indices of a window centered at ``pos``.
        """
        low = pos - self.before
        high = pos + self.after
        return sorted(tuple(set([low, pos, high])))


__all__ = ["Window", "DenseWindow", "SparseWindow"]
