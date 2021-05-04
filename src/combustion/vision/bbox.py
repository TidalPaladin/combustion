#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Dict, Iterable, List, Optional, Tuple, Union

import torch
from numpy import ndarray
from torch import Tensor

from combustion.util import check_dimension, check_dimension_match, check_is_array, check_is_tensor, check_ndim_match

from .convert import to_8bit


PAD_VALUE: float = -1

try:
    import cv2
except ImportError:

    class cv2:
        def __getattr__(self, attr):
            raise ImportError(
                "Bounding box visualization requires cv2. "
                "Please install combustion with 'vision' extras using "
                "pip install combustion [vision]"
            )

        def __setattr__(self, attr):
            raise ImportError(
                "Bounding box visualization requires cv2. "
                "Please install combustion with 'vision' extras using "
                "pip install combustion [vision]"
            )


CORRECT_BOX_COLOR = BOX_COLOR = (255, 0, 0)
TEXT_COLOR = (255, 255, 255)


def visualize_bbox(
    img: Union[Tensor, ndarray],
    bbox: Union[Tensor, ndarray],
    classes: Optional[Union[Tensor, ndarray]] = None,
    scores: Optional[Union[Tensor, ndarray]] = None,
    class_names: Optional[Dict[int, str]] = None,
    box_color: Tuple[int, int, int] = (255, 0, 0),
    text_color: Tuple[int, int, int] = (255, 255, 255),
    label_alpha: float = 0.4,
    thickness: int = 2,
    pad_value: float = -1,
) -> Tensor:
    r"""Adds bounding box visualization to an input array

    Args:
        img (Tensor or numpy.ndarray):
            Background image

        bbox (Tensor or numpy.ndarray):
            Anchor boxes to draw

        classes (Tensor or numpy.ndarray, optional):
            Class labels associated with each anchor box

        scores (Tensor or numpy.ndarray, optional):
            Class scores associated with each anchor box

        class_names (dict, optional):
            Dictionary mapping integer class labels to string names.
            If ``label`` is supplied but ``class_names`` is not, integer
            class labels will be used.

        box_color (tuple of ints, optional):
            A 3-tuple giving the RGB color value to use for anchor boxes.

        text_color (tuple of ints, optional):
            A 3-tuple giving the RGB color value to use for labels.

        label_alpha (float, optional):
            Alpha to apply to the colored background for class labels.

        thickness (int, optional):
            Specifies the thickness of anchor boxes.

        pad_value (float, optional):
            The padding value used when batching boxes and labels

    Returns:
        :class:`torch.Tensor` or :class:`numpy.ndarray` (depending on what was given for `img`)
        with the output image.

    Shape:
        * ``img`` - :math:`(B, C, H, W)` or :math:`(C, H, W)` or :math:`(H, W)`
        * ``bbox`` - :math:`(B, N, 4)` or :math:`(N, 4)`
        * ``classes`` - :math:`(B, N, 1)` or :math:`(N, 1)`
        * ``scores`` - :math:`(B, N, S)` or :math:`(N, S)`
        *  Output - same as ``img``
    """
    # type check
    check_is_array(img, "img")
    check_is_array(bbox, "bbox")
    classes is None or check_is_array(classes, "classes")
    scores is None or check_is_array(scores, "scores")

    # ndim check
    classes is None or check_ndim_match(bbox, classes, "bbox", "classes")
    scores is None or check_ndim_match(bbox, scores, "bbox", "scores")

    # more ndim checks, ensure if one input is batched then all inputs are batched
    boxes_batched = bbox.ndim == 3
    img_batched = img.ndim == 4
    if img_batched != boxes_batched:
        raise ValueError(f"Expected bbox.ndim == 3 when img.ndim == 4, found {bbox.shape}, {img.shape}")
    if boxes_batched:
        if classes is not None and classes.ndim != 3:
            raise ValueError(f"Expected classes.ndim == 3, found {classes.ndim}")
        if scores is not None and scores.ndim != 3:
            raise ValueError(f"Expected scores.ndim == 3, found {scores.ndim}")
    batched = img_batched

    # individual dimension checks
    check_dimension(bbox, dim=-1, size=4, name="bbox")
    classes is None or check_dimension(classes, dim=-1, size=1, name="classes")
    classes is None or check_dimension_match(bbox, classes, -2, "bbox", "classes")
    scores is None or check_dimension_match(bbox, scores, -2, "bbox", "scores")
    img_shape = img.shape[-2:]

    # convert to cpu tensor
    img, bbox, classes, scores = [
        torch.as_tensor(x).cpu() if x is not None else x for x in (img, bbox, classes, scores)
    ]

    # add a channel dimension to img if not present
    if img.ndim == 2:
        img = img.view(1, *img.shape)

    # add a batch dimension if not present
    img = img.view(1, *img.shape) if not batched else img
    bbox = bbox.view(1, *bbox.shape) if not batched else bbox
    if classes is not None:
        classes = classes.unsqueeze(0) if not batched else classes
    if scores is not None:
        scores = scores.unsqueeze(0) if not batched else scores

    # convert image to 8-bit and convert to channels_last
    img_was_float = img.is_floating_point()
    img = to_8bit(img.clone(), per_channel=False, same_on_batch=True)
    img = img.permute(0, 2, 3, 1).contiguous()

    # convert img to color if grayscale input
    if img.shape[-1] == 1:
        img = img.repeat(1, 1, 1, 3)

    # get box indices that arent padding
    valid_indices = (bbox == pad_value).all(dim=-1).logical_not_()

    # iterate over each batch, building bbox overlay
    result = []
    batch_size = bbox.shape[0]
    for batch_idx in range(batch_size):
        # if this fails with cryptic cv errors, ensure that img is contiguous
        result_i = img[batch_idx].numpy()

        # extract valid boxes for this batch
        valid_indices_i = valid_indices[batch_idx]
        bbox_i = bbox[batch_idx][valid_indices_i]
        scores_i = scores[batch_idx][valid_indices_i] if scores is not None else None
        classes_i = classes[batch_idx][valid_indices_i] if classes is not None else None

        # loop over each box and draw the annotation onto result_i
        for box_idx, coords in enumerate(bbox_i):
            x_min, y_min, x_max, y_max = [int(c) for c in coords]

            # draw the bounding box
            cv2.rectangle(
                result_i,
                (x_min, y_min),
                (x_max, y_max),
                box_color,
                thickness,
            )

            # add class labels to bounding box text if present
            text = ""
            if classes_i is not None:
                cls = int(classes_i[box_idx].item())
                # use class integer -> str name if mapping is given, otherwise use class integer
                if class_names is not None:
                    text += class_names.get(cls, f"Class {cls}")
                else:
                    text += f"Class {cls}"

            # add score labels to bounding box text if present
            if scores_i is not None:
                if classes_i is not None:
                    text += " - "
                # add the first score
                text += f"{scores_i[box_idx, 0].item():0.3f}"
                # if multiple scores are present, add those
                num_scores = scores_i.shape[-1]
                for score_idx in range(1, num_scores):
                    text += f" | {scores_i[box_idx, score_idx].item():0.3f}"

            # tag bounding box with class name / integer id
            ((text_width, text_height), _) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
            cv2.rectangle(result_i, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), box_color, -1)
            cv2.putText(
                result_i,
                text,
                (x_min, y_min - int(0.3 * text_height)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                text_color,
                lineType=cv2.LINE_AA,
            )

        # permute back to channels first and add to result list
        result_i = torch.from_numpy(result_i).permute(-1, 0, 1)
        result.append(result_i)

    if len(result) > 1:
        result = torch.stack(result, dim=0)
    else:
        result = result[0]

    # ensure we include a batch dim if one was present in inputs
    if batched and batch_size == 1:
        result = result.view(1, *result.shape)

    if img_was_float:
        result = result.float().div_(255)

    return result


def split_box_target(target: Tensor, split_label: Union[bool, Iterable[int]] = False) -> Tuple[Tensor, ...]:
    r"""Split a bounding box label set into box coordinates and label tensors.

    .. note::
        This operation returns views of the original tensor.

    Args:
        target (:class:`torch.Tensor`):
            The target to split.

        split_label (bool or iterable of ints):
            Whether to further decompose the label tensor. If ``split_label`` is ``True``, split
            the label tensor along the last dimension. If an interable of ints is given, treat each
            int as a split size arugment to :func:`torch.split` along the last dimension.

    Shape:
        * ``target`` - :math:`(*, N, 4 + C)` where :math:`N` is the number of boxes and :math:`C` is the
          number of labels associated with each box.

        * Output - :math:`(*, N, 4)` and :math:`(*, N, C)`
    """
    check_is_tensor(target, "target")
    bbox = target[..., :4]
    label = target[..., 4:]

    if isinstance(split_label, bool) and not split_label:
        return bbox, label

    num_labels = label.shape[-1]

    # setup split size of 1 if bool given
    if isinstance(split_label, bool):
        split_label = [
            1,
        ] * num_labels

    lower_bound = 0
    upper_bound = 0
    final_label = []
    for delta in split_label:
        upper_bound = lower_bound + delta
        final_label.append(label[..., lower_bound:upper_bound])
        lower_bound = upper_bound
    assert len(final_label) == len(split_label)
    return tuple([bbox] + final_label)


def split_bbox_scores_class(target: Tensor, split_scores: Union[bool, Iterable[int]] = False) -> Tuple[Tensor, ...]:
    r"""Split a predicted bounding box into box coordinates, probability score, and predicted class.
    This implementation supports multiple score assignments for each box. It is expected that ``target``
    be ordered along the last dimension as ``bbox``, ``scores``, ``class``.

    .. note::
        This operation returns views of the original tensor.

    Args:
        target (:class:`torch.Tensor`):
            The target to split.

        split_scores (bool or iterable of ints):
            Whether to further decompose the scores tensor. If ``split_scores`` is ``True``, split
            the scores tensor along the last dimension. If an interable of ints is given, treat each
            int as a split size arugment to :func:`torch.split` along the last dimension.

    Shape:
        * ``target`` - :math:`(*, N, 4 + S + 1)` where :math:`N` is the number of boxes and :math:`S` is the
          number of scores associated with each box.
        * Output - :math:`(*, N, 4)`, :math:`(*, N, S)`, and :math:`(*, N, 1)`
    """
    check_is_tensor(target, "target")
    bbox = target[..., :4]
    scores = target[..., 4:-1]
    cls = target[..., -1:]

    if isinstance(split_scores, bool) and not split_scores:
        return bbox, scores, cls

    num_scores = scores.shape[-1]

    # setup split size of 1 if bool given
    if isinstance(split_scores, bool):
        split_scores = [
            1,
        ] * num_scores

    lower_bound = 0
    upper_bound = 0
    final_scores = []
    for delta in split_scores:
        upper_bound = lower_bound + delta
        final_scores.append(scores[..., lower_bound:upper_bound])
        lower_bound = upper_bound
    assert len(final_scores) == len(split_scores)
    return tuple([bbox] + final_scores + [cls])


def combine_box_target(bbox: Tensor, label: Tensor, *extra_labels) -> Tensor:
    r"""Combine a bounding box coordinates and labels into a single label.

    Args:
        bbox (:class:`torch.Tensor`):
            Coordinates of the bounding box.

        label (:class:`torch.Tensor`):
            Label associated with each bounding box.

    Shape:
        * ``bbox`` - :math:`(*, N, 4)`
        * ``label`` - :math:`(*, N, 1)`
        * Output - :math:`(*, N, 4 + 1)`
    """
    # input validation
    check_is_tensor(bbox, "bbox")
    check_is_tensor(label, "label")
    if bbox.shape[-1] != 4:
        raise ValueError(f"Expected bbox.shape[-1] == 4, found shape {bbox.shape}")
    if bbox.shape[:-1] != label.shape[:-1]:
        raise ValueError(f"Expected bbox.shape[:-1] == label.shape[:-1], found shapes {bbox.shape}, {label.shape}")

    return torch.cat([bbox, label, *extra_labels], dim=-1)


def combine_bbox_scores_class(bbox: Tensor, cls: Tensor, scores: Tensor, *extra_scores) -> Tensor:
    r"""Combine a bounding box coordinates and labels into a single label. Combined tensor
    will be ordered along the last dimension as ``bbox``, ``scores``, ``cls``.

    Args:
        bbox (:class:`torch.Tensor`):
            Coordinates of the bounding box.

        cls (:class:`torch.Tensor`):
            Class associated with each bounding box

        scores (:class:`torch.Tensor`):
            Probability associated with each bounding box.

        *extra_scores (:class:`torch.Tensor`):
            Additional scores to combine

    Shape:
        * ``bbox`` - :math:`(*, N, 4)`
        * ``scores`` - :math:`(*, N, S)`
        * ``cls`` - :math:`(*, N, 1)`
        * Output - :math:`(*, N, 4 + S + 1)`
    """
    # input validation
    check_is_tensor(bbox, "bbox")
    check_is_tensor(scores, "scores")
    check_is_tensor(cls, "cls")
    if bbox.shape[-1] != 4:
        raise ValueError(f"Expected bbox.shape[-1] == 4, found shape {bbox.shape}")
    return torch.cat([bbox, scores, *extra_scores, cls], dim=-1)


def batch_box_target(target: List[Tensor], pad_value: float = -1) -> Tensor:
    r"""Combine multiple distinct bounding box targets into a single batched target.

    Args:
        target (list of :class:`torch.Tensor`):
            List of bounding box targets to combine

        pad_value (float):
            Padding value to use when creating the batch

    Shape:
        * ``target`` - :math:`(*, N_i, 4 + C)` where :math:`N_i` is the number of boxes and :math:`C` is the
          number of labels associated with each box.

        * Output - :math:`(B, N, 4 + c)`
    """
    max_boxes = 0
    for elem in target:
        # check_is_tensor(elem, "target_elem")
        max_boxes = max(max_boxes, elem.shape[-2])

    # add a batch dim if not present
    target = [x.unsqueeze(0) if x.ndim < 3 else x for x in target]

    # compute output batch size
    batch_size = torch.tensor([x.shape[0] for x in target]).sum().item()

    # create empty output tensor of correct shape
    output_shape = (batch_size, max_boxes, target[0].shape[-1])
    batch = torch.empty(*output_shape, device=target[0].device, dtype=target[0].dtype).fill_(pad_value)

    # fill output tensor
    start = 0
    for elem in target:
        end = start + elem.shape[0]
        batch[start:end, : elem.shape[-2], :] = elem
        start += elem.shape[0]

    return batch


def unbatch_box_target(target: Tensor, pad_value: float = -1) -> List[Tensor]:
    r"""Splits a padded batch of bounding boxtarget tensors into a list of unpadded target tensors

    Args:
        target (:class:`torch.Tensor`):
            Batch of bounding box targets to split

        pad_value (float):
            Value used for padding when creating the batch

    Shape:
        * ``target`` - :math:`(B, N, 4 + C)` where :math:`N` is the number of boxes and :math:`C` is the
          number of labels associated with each box.

        * Output - :math:`(N, 4 + C)`
    """
    check_is_tensor(target, "target")

    padding_indices = (target == pad_value).all(dim=-1)
    non_padded_coords = (~padding_indices).nonzero(as_tuple=True)

    flat_result = target[non_padded_coords]
    split_size = non_padded_coords[0].unique(return_counts=True)[1]
    return torch.split(flat_result, split_size.tolist(), dim=0)


def flatten_box_target(target: Tensor, pad_value: float = -1) -> List[Tensor]:
    r"""Flattens a batch of bounding box target tensors, removing padded locations

    Args:
        target (:class:`torch.Tensor`):
            Batch of bounding box targets to split

        pad_value (float):
            Value used for padding when creating the batch

    Shape:
        * ``target`` - :math:`(B, N, 4 + C)` where :math:`N` is the number of boxes and :math:`C` is the
          number of labels associated with each box.

        * Output - :math:`(N_{tot}, 4 + C)`
    """
    check_is_tensor(target, "target")
    padding_indices = (target == pad_value).all(dim=-1)
    non_padded_coords = (~padding_indices).nonzero(as_tuple=True)
    return target[non_padded_coords]


def append_bbox_label(old_label: Tensor, new_label: Tensor) -> Tensor:
    r"""Adds a new label element to an existing bounding box target.
    The new label will be concatenated to the end of the last dimension in
    ``old_label``.

    Args:
        old_label (:class:`torch.Tensor`):
            The existing bounding box label

        new_label (:class:`torch.Tensor`):
            The label entry to add to ``old_label``

    Shape:
        * ``old_label`` - :math:`(*, N, C_0)`
        * ``new_label`` - :math:`(B, N, C_1`)`
        * Output - :math:`(B, N, C_0 + C_1)`
    """
    check_is_tensor(old_label, "old_label")
    check_is_tensor(new_label, "new_label")
    check_ndim_match(old_label, new_label, "old_label", "new_label")
    if old_label.shape[:-1] != new_label.shape[:-1]:
        raise ValueError(
            "expected old_label.shape[:-1] == new_label.shape[:-1], " "found {old_label.shape}, {new_label.shape}"
        )

    return torch.cat([old_label, new_label], dim=-1)


def filter_bbox_classes(
    target: Tensor, keep_classes: Iterable[int], pad_value: float = -1, return_inverse: bool = False
) -> Tensor:
    r"""Filters bounding boxes based on class, replacing bounding boxes that do not meet the criteria
    with padding. Integer class ids should be the last column in ``target``.

    Args:
        target (:class:`torch.Tensor`):
            Bounding boxes to filter

        keep_classes (iterable of ints):
            Integer id of the classes to keep

        pad_value (float):
            Value used to indicate padding in both input and output tensors

        return_inverse (:class:`torch.Tensor`):
            If ``True``, remove boxes with classes not in ``keep_classes``

    Shape:
        * ``target`` - :math:`(*, N, C)`
        * Output - same as ``target``
    """
    check_is_tensor(target, "target")
    if not isinstance(keep_classes, Iterable):
        raise TypeError(f"Expected iterable for keep_classes, found {type(keep_classes)}")
    if not keep_classes:
        raise ValueError(f"Expected non-empty iterable for keep classes, found {keep_classes}")

    locations_to_keep = torch.zeros_like(target[..., -1]).bool()
    for keep_cls in keep_classes:
        if not isinstance(keep_cls, (float, int)):
            raise TypeError(f"Expected int or float for keep_classes elements, found {type(keep_cls)}")
        locations_for_cls = torch.as_tensor(target[..., -1] == keep_cls)
        locations_to_keep.logical_or_(locations_for_cls)

    if return_inverse:
        locations_to_keep.logical_not_()

    target = target.clone()
    target[~locations_to_keep] = -1
    return target
