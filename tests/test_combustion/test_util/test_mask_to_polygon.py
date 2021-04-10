#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import torch

from combustion.util import mask_to_box, mask_to_instances, mask_to_polygon


@pytest.fixture
def case():
    mask = torch.tensor(
        [
            [1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0],
        ]
    )
    instances = torch.tensor(
        [
            [1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 2, 0],
            [0, 0, 0, 0, 2, 2, 2],
            [0, 0, 0, 0, 2, 2, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [3, 3, 3, 0, 0, 0, 0],
        ]
    )
    mask = torch.tensor(
        [
            [1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0],
        ]
    )
    boxes = torch.tensor(
        [
            [0, 0, 2, 2],
            [4, 4, 6, 6],
            [8, 0, 8, 2],
        ]
    )
    polygons = [
        torch.tensor(
            [
                [0, 0],
                [0, 1],
                [0, 2],
                [1, 0],
                [1, 2],
                [2, 1],
                [2, 2],
            ]
        ),
        torch.tensor(
            [
                [4, 5],
                [5, 4],
                [5, 5],
                [5, 6],
                [6, 4],
                [6, 5],
            ]
        ),
        torch.tensor(
            [
                [8, 0],
                [8, 1],
                [8, 2],
            ]
        ),
    ]
    return {"mask": mask, "boxes": boxes, "polygons": polygons, "instances": instances}


def test_mask_to_instances(case):
    instances = mask_to_instances(case["mask"])
    true_instances = case["instances"]
    assert torch.allclose(instances, true_instances)


def test_mask_to_box(case):
    boxes = mask_to_box(case["mask"])
    true_boxes = case["boxes"]
    assert torch.allclose(boxes, true_boxes)


def test_mask_to_polygon(case):
    polygons = mask_to_polygon(case["mask"])
    true_polygons = case["polygons"]

    assert len(polygons) == len(true_polygons)
    for p, p_true in zip(polygons, true_polygons):
        assert torch.allclose(p, p_true)
