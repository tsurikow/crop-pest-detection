from __future__ import annotations

import torch
from torch import nn


class TritonDetectorWrapper(nn.Module):
    def __init__(
        self, detector: nn.Module, max_dets: int = 100, score_thr: float = 0.05
    ):
        super().__init__()
        self.detector = detector
        self.max_dets = int(max_dets)
        self.score_thr = float(score_thr)

    def forward(self, image: torch.Tensor):
        out = self.detector([image])[0]
        boxes = out["boxes"]
        scores = out["scores"]
        labels = out["labels"]

        if self.score_thr > 0.0:
            keep = scores >= self.score_thr
            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]

        if torch.numel(scores) > 0:
            order = torch.argsort(scores, descending=True)
            order = order[: self.max_dets]
            boxes = boxes.index_select(0, order)
            scores = scores.index_select(0, order)
            labels = labels.index_select(0, order)

        num = torch._shape_as_tensor(scores)[0:1]

        boxes_out = torch.zeros(
            (self.max_dets, 4), dtype=boxes.dtype, device=boxes.device
        )
        scores_out = torch.zeros(
            (self.max_dets,), dtype=scores.dtype, device=scores.device
        )
        labels_out = torch.zeros(
            (self.max_dets,), dtype=torch.int64, device=labels.device
        )

        n = torch._shape_as_tensor(scores)[0]
        idx = torch.arange(n, dtype=torch.int64, device=scores.device)

        boxes_out = boxes_out.index_copy(0, idx, boxes)
        scores_out = scores_out.index_copy(0, idx, scores)
        labels_out = labels_out.index_copy(0, idx, labels.to(torch.int64))

        return boxes_out, scores_out, labels_out, num
