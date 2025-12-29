from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def _xyxy_to_xywh(box: np.ndarray) -> List[float]:
    x1, y1, x2, y2 = box.tolist()
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    return [float(x1), float(y1), float(w), float(h)]


@torch.no_grad()
def coco_map_from_predictions(
    preds: List[Dict[str, torch.Tensor]],
    targets: List[Dict[str, Any]],
    num_classes: int,
) -> Dict[str, float]:
    """
    preds: list of dicts from torchvision detector in eval mode:
      {"boxes": [N,4] xyxy, "labels": [N], "scores":[N]}

    targets: list of dicts from dataset:
      {"boxes": [M,4] xyxy, "labels":[M], "image_id": tensor([id])}

    num_classes: number of foreground classes (12)
    """
    images = []
    annotations = []
    results = []

    ann_id = 1

    for p, t in zip(preds, targets):
        image_id = int(t["image_id"].item()) if torch.is_tensor(t["image_id"]) else int(t["image_id"])
        width = None
        height = None
        if "image_size" in t:
            height, width = int(t["image_size"][0]), int(t["image_size"][1])

        img_rec = {"id": image_id}
        if width is not None and height is not None:
            img_rec["width"] = width
            img_rec["height"] = height
        images.append(img_rec)

        gt_boxes = t["boxes"].detach().cpu().numpy().astype(np.float32)
        gt_labels = t["labels"].detach().cpu().numpy().astype(np.int64)

        for box, label in zip(gt_boxes, gt_labels):
            bbox_xywh = _xyxy_to_xywh(box)
            area = float(max(0.0, bbox_xywh[2]) * max(0.0, bbox_xywh[3]))
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": int(label),
                    "bbox": bbox_xywh,
                    "area": area,
                    "iscrowd": 0,
                }
            )
            ann_id += 1

        pr_boxes = p["boxes"].detach().cpu().numpy().astype(np.float32)
        pr_labels = p["labels"].detach().cpu().numpy().astype(np.int64)
        pr_scores = p["scores"].detach().cpu().numpy().astype(np.float32)

        for box, label, score in zip(pr_boxes, pr_labels, pr_scores):
            results.append(
                {
                    "image_id": image_id,
                    "category_id": int(label),
                    "bbox": _xyxy_to_xywh(box),
                    "score": float(score),
                }
            )

    coco_dict = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": i, "name": str(i)} for i in range(1, num_classes + 1)],
    }

    coco_gt = COCO()
    coco_gt.dataset = coco_dict
    coco_gt.createIndex()

    if len(results) == 0:
        return {"mAP": 0.0, "mAP50": 0.0}

    coco_dt = coco_gt.loadRes(results)

    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats = coco_eval.stats
    return {
        "mAP": float(stats[0]),
        "mAP50": float(stats[1]),
    }