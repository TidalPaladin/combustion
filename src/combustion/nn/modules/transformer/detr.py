#!/usr/bin/env python
# -*- coding: utf-8 -*-



import torch.nn.functional as F
import torch
from torch import Tensor
from scipy.optimize import linear_sum_assignment
from torch import nn
from typing import Final, List, Tuple, Iterator, Optional, TypeVar, Callable
from dataclasses import dataclass

from combustion.nn.loss import CompleteIoULoss, CategoricalFocalLoss
from combustion.nn.loss.ciou import complete_iou_loss
from .perceiver import PerceiverLayer
from .common import MLP
from .position import FourierLogspace, LearnableFourierFeatures
from combustion.nn.loss.fcos import assign_boxes_to_levels, DEFAULT_INTEREST_RANGE
from combustion.util.dataclasses import TensorDataclass

PAD_VAL: Final = -1

T = TypeVar("T")
#Transform = Annotated[Callable[[T], T]]
Transform = Callable


@dataclass(repr=False)
class DETRPrediction(TensorDataclass):
    logits: Tensor
    coords: Tensor

    def __post_init__(self):
        # fp16 will cause problems so cast to float
        object.__setattr__(self, "logits", self.logits.float())
        object.__setattr__(self, "coords", self.coords.float())

    def __getitem__(self, idx: int) -> "DETRPrediction":
        assert self.is_batched
        return self.replace(logits=self.logits[idx], coords=self.coords[idx])

    @property
    def xyxy_coords(self) -> Tensor:
        cxcy = self.coords[..., :2]
        wh = self.coords[..., -2:]
        return torch.cat([cxcy - wh/2, cxcy + wh/2], dim=-1)

    @property
    def xywh_coords(self) -> Tensor:
        return self.coords

    @property
    def is_batched(self) -> bool:
        return self.logits.ndim == 3

    def scale(self, level: int) -> "DETRPrediction":
        coords = self.coords.clone()
        coords[..., 2:] = coords[..., 2:] * 2 ** level
        return self.replace(coords=coords)

    @property
    def num_classes(self) -> int:
        return self.logits.shape[-1]

    @property
    def scores(self) -> Tensor:
        return self.logits.softmax(dim=-1)

    @property
    def classes(self) -> Tensor:
        return self.logits.argmax(dim=-1, keepdim=True)

    @property
    def max_score(self) -> Tensor:
        return self.scores.amax(dim=-2)

    @property
    def max_logits(self) -> Tensor:
        return self.logits.amax(dim=-2)

    @property
    def is_empty_box(self) -> Tensor:
        return self.classes.squeeze(-1) == self.num_classes - 1

    def postprocess(self) -> "DETRPrediction":
        keep = self.is_empty_box.logical_not_()
        coords = self.coords[keep]
        logits = self.logits[keep]
        return self.replace(logits=logits, coords=coords)


@dataclass(repr=False)
class DETRTarget(TensorDataclass):
    classes: Tensor
    coords: Tensor

    @property
    def xyxy_coords(self) -> Tensor:
        return self.coords

    @property
    def xywh_coords(self) -> Tensor:
        x1y1 = self.coords[..., :2]
        x2y2 = self.coords[..., 2:]
        cxcy = (x1y1 + x2y2) / 2
        wh = (x2y2 - x1y1)
        result = torch.cat([cxcy, wh], dim=-1)
        result[~self.valid_targets] = -1
        return result

    def __add__(self, other: "DETRTarget") -> "DETRTarget":
        assert isinstance(other, DETRTarget)
        classes = torch.where(self.valid_targets, self.classes, other.classes)
        coords = torch.where(self.valid_targets, self.coords, other.coords)
        return self.replace(classes=classes, coords=coords)

    @property
    def valid_targets(self) -> Tensor:
        return (self.coords == -1).all(dim=-1).logical_not_()

    @property
    def num_valid_targets(self) -> int:
        return int(self.valid_targets.sum().item())

    def drop_invalid_targets(self, x: Tensor) -> Tensor:
        return x[self.valid_targets]

    @classmethod
    def from_boxes(
        cls, 
        bbox: Tensor, 
        classes: Tensor, 
        image_size: Tuple[int, int],
    ):
        Bb, Nb, Cb = bbox.shape
        Bc, Nc, Cc = classes.shape
        assert Bb == Bc
        assert Nb == Nc
        assert Cb == 4
        assert Cc == 1
        B = Bb
        N = Nb

        bbox = bbox.clone()
        classes = classes.clone()

        # boxes to normalized boxes
        padding = (bbox == PAD_VAL).all(dim=-1)
        H, W = image_size
        divisor = bbox.new_tensor([W, H, W, H])
        bbox = bbox / divisor
        bbox[padding] = PAD_VAL

        return cls(classes, bbox)


@dataclass(repr=False)
class MatchingResult(TensorDataclass):
    batch_idx: Tensor
    source_idx: Tensor
    target_idx: Tensor

    def __eq__(self, other: "MatchingResult") -> bool:
        return bool(
            (self.batch_idx == other.batch_idx)
            .logical_and_(self.source_idx == other.source_idx)
            .logical_and_(self.target_idx == other.target_idx)
            .all()
        )

    def __post_init__(self):
        assert self.batch_idx.numel() == self.source_idx.numel() == self.target_idx.numel()

    def apply(self, pred: DETRPrediction, true: DETRTarget) -> DETRTarget:
        # the default target, with all GT as empty boxes
        final_target = self.make_default_target(proto=pred)

        # the updated target, with all GT as empty boxes
        final_target.coords[self.batch_idx, self.source_idx] = true.coords[self.batch_idx, self.target_idx].type_as(final_target.coords)
        final_target.classes[self.batch_idx, self.source_idx] = true.classes[self.batch_idx, self.target_idx].type_as(final_target.classes)

        return final_target

    def make_default_target(self, proto: DETRPrediction) -> DETRTarget:
        num_classes = proto.num_classes
        coords = torch.full_like(proto.coords, -1, dtype=torch.float)
        classes = torch.full_like(proto.logits[..., 0, None], num_classes - 1, dtype=torch.long)
        return DETRTarget(classes, coords)
    
    @classmethod
    def default_matching(cls, device="cpu") -> "MatchingResult":
        return cls(
            torch.empty(0, dtype=torch.long, device=device),
            torch.empty(0, dtype=torch.long, device=device),
            torch.empty(0, dtype=torch.long, device=device),
        )

    @classmethod
    def from_cost_matrix(cls, cost_matrix: Tensor):
        B, S, T = cost_matrix.shape
        sources: List[Tensor] = []
        targets: List[Tensor] = []
        batches: List[Tensor] = []

        device = cost_matrix.device
        cost_matrix = cost_matrix.cpu()
        padding = cost_matrix.isinf().all(dim=-2, keepdim=False)
        for i, (pad, mat) in enumerate(zip(padding, cost_matrix)):
            if pad.all():
                continue

            assert not mat[..., ~pad].isinf().any()
            assert not mat[..., ~pad].isnan().any()

            _src, _tgt = linear_sum_assignment(mat[..., ~pad])

            src = torch.as_tensor(_src, dtype=torch.long, device=device)
            tgt = torch.as_tensor(_tgt, dtype=torch.long, device=device)

            # unpadded target indices must be mapped back to padded indices
            idx = torch.arange(mat.shape[-1], device=device)
            tgt = idx[~pad][tgt]

            batch_idx = torch.full_like(src, i)
            sources.append(src)
            targets.append(tgt)
            batches.append(batch_idx)

        final_batches = torch.cat(batches) if batches else cost_matrix.new_empty(0, dtype=torch.long, device=device)
        final_sources = torch.cat(sources) if sources else cost_matrix.new_empty(0, dtype=torch.long, device=device)
        final_targets = torch.cat(targets) if targets else cost_matrix.new_empty(0, dtype=torch.long, device=device)
            
        return cls(
            final_batches,
            final_sources,
            final_targets,
        )



class HungarianMatcher:
    # https://github.com/facebookresearch/detr/blob/main/models/matcher.py
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_dist: float = 1):
        super().__init__()
        self.cost_class = cost_class
        self.cost_dist = cost_dist
        self.ciou = CompleteIoULoss(reduction="none")
        assert cost_class != 0 or cost_dist != 0, "all costs cant be 0"

    def compute_class_cost(self, pred: Tensor, true: Tensor) -> Tensor:
        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        pred, true = self.match_shapes(pred, true)
        B, Np, Nt, C = pred.shape
        B, Np, Nt, _ = true.shape

        pred_idx = torch.arange(C, device=pred.device).view(1, 1, 1, C).expand_as(pred)
        result = torch.where(pred_idx == true, -pred, pred.new_tensor(float("inf"))).amin(dim=-1)
        assert result.shape == (B, Np, Nt)
        return result

    @staticmethod
    def match_shapes(pred: Tensor, true: Tensor) -> Tuple[Tensor, Tensor]:
        Bp, Np, _ = pred.shape
        Bt, Nt, _ = true.shape
        assert Bp == Bt
        B = Bp
        pred = pred.view(B, Np, 1, -1).expand(-1, -1, Nt, -1).contiguous()
        true = true.view(B, 1, Nt, -1).expand(-1, Np, -1, -1).contiguous()
        return pred, true

    def compute_distance_cost(self, pred: Tensor, true: Tensor) -> Tensor:
        pred, true = self.match_shapes(pred, true)
        B, Np, Nt, _ = pred.shape
        B, Np, Nt, _ = true.shape

        cost = pred.new_full((B, Np, Nt), fill_value=float("inf"))
        padding = (true == PAD_VAL).all(dim=-1)

        cost[~padding] = complete_iou_loss(pred[~padding], true[~padding], reduction="none")

        assert cost.shape == (B, Np, Nt)
        return cost

    @torch.no_grad()
    def __call__(
        self,
        pred_logits: Tensor,
        pred_boxes: Tensor,
        classes: Tensor,
        true_boxes: Tensor
    ) -> MatchingResult:
        Bp = pred_logits.shape[0]
        Bt = classes.shape[0]
        assert Bp == Bt

        if not classes.numel():
            return MatchingResult.default_matching(device=classes.device)

        pred_scores = pred_logits.float().softmax(-1)  
        cost_class = self.compute_class_cost(pred_scores, classes)
        cost_dist = self.compute_distance_cost(pred_boxes, true_boxes)

        # Final cost matrix
        cost_matrix = self.cost_dist * cost_dist + self.cost_class * cost_class
        matching = MatchingResult.from_cost_matrix(cost_matrix)
        return matching


class DETRHead(nn.Module):

    def __init__(
        self, 
        latent_d: int, 
        input_d: int,
        num_boxes: int,
        num_layers: int,
        num_classes: int,
        **kwargs
    ):
        super().__init__()
        latent_d = input_d
        self.query = nn.Parameter(torch.randn(num_boxes, 1, latent_d))
        self.scale = nn.Embedding(5, input_d)

        block = PerceiverLayer(
            latent_d,
            input_d,
            latent_l=None,
            nhead_input=4,
            **kwargs,
        )
        self.encoder = nn.ModuleList([block for _ in range(num_layers)])
        encoder = nn.TransformerDecoderLayer(latent_d, nhead=4, dim_feedforward=latent_d)
        self.encoder = nn.TransformerDecoder(encoder, 6)
        #for _ in range(num_layers):
            #self.encoder.append(block1.duplicate())
        #self.decoder = nn.ModuleList(
        #    block2.duplicate()
        #    for _ in range(num_layers)
        #)

        #block = nn.TransformerDecoderLayer(latent_d, nhead=8, dim_feedforward=512)
        #self.transformer = nn.TransformerDecoder(block, num_layers)

        self.pos_enc = FourierLogspace(2, input_d, 512, 32, zero_one_norm=False)

        class_neck = MLP(latent_d, latent_d, latent_d, dropout=0.1)
        box_neck = MLP(latent_d, latent_d, latent_d, dropout=0.1)

        class_head = nn.Linear(latent_d, num_classes+1)
        box_head = nn.Linear(latent_d, 4)

        # class init
        null_prior = 0.8
        fill_value = torch.tensor((1 - null_prior) / num_classes).logit().item()
        class_bias = torch.empty(num_classes+1).fill_(fill_value)
        class_bias[-1] = torch.tensor(null_prior).logit()
        class_head.bias = nn.Parameter(class_bias)

        # box init
        box_head.bias = nn.Parameter(torch.tensor([0.5, 0.5, 0.2, 0.2]).logit())
        nn.init.normal_(box_head.weight, 0, 0.1)

        self.class_head = nn.Sequential(
            class_neck,
            class_head,
        )
        self.box_head = nn.Sequential(
            box_neck,
            box_head,
            nn.Sigmoid()
        )
        

    def forward(self, fpn: List[Tensor]) -> DETRPrediction:
        N = fpn[0].shape[0]
        latent = self.query.expand(-1, N, -1)
        for i, fpn_level in enumerate(fpn):
            N, D, H, W = fpn_level.shape
            L = H*W

            grid = LearnableFourierFeatures.from_grid((H, W), proto=fpn_level)
            pos_enc = self.pos_enc(grid).view(L, 1, D).expand(-1, N, -1)
            scale = self.scale(torch.full_like(pos_enc[..., 0], i, dtype=torch.long))
            fpn_level = fpn_level.view(N, D, L).movedim(-1, 0)
            fpn_level += (pos_enc + scale)

            #for layer in self.encoder:
            #    fpn_level, latent = layer(fpn_level, latent)

            latent = self.encoder(latent, fpn_level)

        #query = self.query.expand(-1, N, -1)
        #for layer in self.decoder:
        #    query, latent = layer(query, latent)
        query = latent

        boxes = self.box_head(query).movedim(1, 0)
        scores = self.class_head(query).movedim(1, 0)
        result = DETRPrediction(scores, boxes)
        return result



@dataclass(repr=False)
class DETRLoss(TensorDataclass):
    score_loss: Tensor
    coord_loss: Tensor
    matching_loss: Tensor
    cardinality_loss: Tensor

    @property
    def total_loss(self) -> Tensor:
        return self.score_loss + self.coord_loss

    def __add__(self, other: "DETRLoss") -> "DETRLoss":
        assert isinstance(other, DETRLoss)
        score_loss = self.score_loss + other.score_loss
        coord_loss = self.coord_loss + other.coord_loss
        return self.replace(score_loss=score_loss, coord_loss=coord_loss)


class DETRCriterion:

    def __init__(
        self, 
        gamma: float = 2, 
        cost_class: float = 1, 
        cost_dist: float = 1, 
        interest_range: Tuple[Tuple[int, int], ...] = DEFAULT_INTEREST_RANGE,
        empty_weight: float = 0.1,
        **kwargs,
    ):
        self.matcher = HungarianMatcher(cost_class, cost_dist)
        self.ciou = CompleteIoULoss()
        self.cls_criterion = CategoricalFocalLoss(gamma, **kwargs)
        self.interest_range = interest_range
        self.empty_weight = empty_weight

    def __call__(
        self, 
        pred: DETRPrediction, 
        classes: Tensor,
        coords: Tensor,
        image_size: Tuple[int, int],
        **kwargs
    ) -> DETRLoss:
        target = DETRTarget.from_boxes(coords, classes, image_size, **kwargs)
        return self.compute_from_prepared(pred, target)

    def compute_from_prepared(
        self, 
        pred: DETRPrediction, 
        true: DETRTarget,
    ) -> DETRLoss:
        num_classes = pred.num_classes
        matching = self.matcher(pred.logits, pred.xywh_coords, true.classes, true.xywh_coords)
        matching_loss = self.compute_matching_loss(pred, true, matching)
        true = matching.apply(pred, true)

        pred_coords = true.drop_invalid_targets(pred.xywh_coords)
        true_coords = true.drop_invalid_targets(true.xywh_coords)

        num_classes = pred.logits.shape[-1]
        empty_weight = pred.logits.new_ones(num_classes, requires_grad=False)
        empty_weight[-1] = self.empty_weight
        cls_loss = F.cross_entropy(pred.logits.reshape(-1, num_classes), true.classes.flatten(), empty_weight) 

        if pred_coords.numel():
            box_loss = complete_iou_loss(pred_coords, true_coords)
        else:
            box_loss = pred_coords.new_tensor(0)

        cardinality_loss = self.compute_cardinality_loss(pred, true)

        loss = DETRLoss(cls_loss, box_loss, matching_loss, cardinality_loss)
        return loss

    @torch.no_grad()
    def compute_cardinality_loss(
        self, 
        pred: DETRPrediction, 
        true: DETRTarget,
    ) -> Tensor:
        num_pred_boxes = pred.is_empty_box.logical_not_().sum()
        num_true_boxes = true.num_valid_targets
        return pred.coords.new_tensor(num_pred_boxes - num_true_boxes).abs_()

    @torch.no_grad()
    def compute_matching_loss(
        self, 
        pred: DETRPrediction, 
        true: DETRTarget,
        matching: MatchingResult,
    ) -> Tensor:
        true = matching.apply(pred, true)
        pred_coords = true.drop_invalid_targets(pred.xywh_coords)
        true_coords = true.drop_invalid_targets(true.xywh_coords)
        result = (pred_coords[..., :2] - true_coords[..., :2]).pow(2).sum(dim=-1).sqrt().mean()
        return result
