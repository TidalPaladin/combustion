#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from argparse import Namespace
from itertools import islice
from typing import Generator, Iterable, Tuple

import torch
from torch import Tensor


class Window(ABC):
    """Helper to apply a window over an iterable or set of indices
    """

    def __init__(self, before: int = 0, after: int = 0):
        """__init__

        :param before: Number of frames before the current frame to include in the window.
        :type before: int
        :param after: Number of frames after the current frame to include in the window.
        :type after: int
        """
        if int(before) < 0:
            raise ValueError("before must be int >= 0")
        if int(after) < 0:
            raise ValueError("before must be int >= 0")
        self.before = int(before)
        self.after = int(after)
        if not (self.before or self.after):
            raise ValueError("before or after must not be 0")

    def __len__(self):
        """__len__
        Returns the number of frames in the window
        """
        return self.before + self.after + 1

    def __call__(self, examples: Iterable) -> Generator[Tuple[Tensor, Tensor], None, None]:
        """__call__
        Generates frames, labels tuples by applying the window to `examples`.
        For an input/labels of shape `CxHxW`, and a window of len `D`, the output will be
        of frame, label tuples of shape (`CxDxHxW`, `CxHxW`).

        :param examples: An iterable of (frame, label) tuples to be windowed
        :type examples: Iterable
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
        """indices
        Given an index `pos`, return a tuple of indices that are part of the window 
        centered at `pos`.

        :param pos: The index of the window center
        :type pos: int
        :rtype: Tuple[int, ...]
        :returns: Indices that are part of the window centered at `pos`
        """
        low = pos - self.before
        high = pos + self.after
        return tuple(set(range(low, high)))

    def estimate_size(self, num_frames: int) -> int:
        """estimate_size
        Given a number of examples in the un-windowed input, estimate the number of
        examples in the windowed result.

        :param num_frames: The number of frames in the un-windowed dataset.
        :type num_frames: int
        :rtype: int
        :returns: Estimated number of frames in the windowed output.
        """
        return num_frames - (self.before + self.after)

    @classmethod
    def from_args(cls, args: Namespace) -> "Window":
        """from_args
        Create a Window based on CLI flags

        :param args: CLI flags
        :type args: Namespace
        :rtype: Window
        """
        if args.dense_window != 0:
            size = args.dense_window
            return DenseWindow(size, size)
        elif args.sparse_window != 0:
            size = args.sparse_window
            return SparseWindow(size, size)
        else:
            raise ValueError("args.dense_window and args.sparse_window were both 0")


class DenseWindow(Window):
    """Window
    Applies a window function

    """

    def indices(self, pos: int) -> Tuple[int, ...]:
        """indices
        Given an index `pos`, return a tuple of indices that are part of the window 
        centered at `pos`.

        :param pos: The index of the window center
        :type pos: int
        :rtype: Tuple[int, ...]
        :returns: Indices that are part of the window centered at `pos`
        """
        low = pos - self.before
        high = pos + self.after
        return tuple(set(range(low, high + 1)))


class SparseWindow(Window):
    """Window
    Applies a window function

    """

    def indices(self, pos: int) -> Tuple[int, ...]:
        """indices
        Given an index `pos`, return a tuple of indices that are part of the window 
        centered at `pos`.

        :param pos: The index of the window center
        :type pos: int
        :rtype: Tuple[int, ...]
        :returns: Indices that are part of the window centered at `pos`
        """
        low = pos - self.before
        high = pos + self.after
        return sorted(tuple(set([low, pos, high])))
