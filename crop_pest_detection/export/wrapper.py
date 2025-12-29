from __future__ import annotations

import torch

class TritonDetectorWrapper(torch.nn.Module):
    def __init__(self, detector: torch.nn.Module, max_dets: int = 100, score_thr: float = 0.05):
        super().__init__()
        self.detector = detector
        self.max_dets = max_dets
        self.score_thr = score_thr

    def forward(self, images: torch.Tensor):
        preds = self.detector([images])
        p = preds[0]

        boxes = p["boxes"]
        scores = p["scores"]
        labels = p["labels"].to(torch.int64)

        keep = scores >= self.score_thr
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        n = boxes.shape[0]
        num = torch.zeros((1,), dtype=torch.int64)
        num[0] = n

        max_dets_t = torch.tensor(self.max_dets, dtype=torch.int64)
        n_t = torch.tensor(n, dtype=torch.int64)
        k_t = torch.minimum(n_t, max_dets_t)
        k = int(k_t.item())

        boxes_out = torch.zeros((self.max_dets, 4), dtype=boxes.dtype)
        scores_out = torch.zeros((self.max_dets,), dtype=scores.dtype)
        labels_out = torch.zeros((self.max_dets,), dtype=torch.int64)

        if k > 0:
            boxes_out[:k] = boxes[:k]
            scores_out[:k] = scores[:k]
            labels_out[:k] = labels[:k]

        return boxes_out, scores_out, labels_out, num