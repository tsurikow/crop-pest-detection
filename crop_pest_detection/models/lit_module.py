from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pytorch_lightning as pl
import torch

from .detector import build_fasterrcnn_resnet50_fpn
from crop_pest_detection.eval.coco_eval import coco_map_from_predictions

class PestDetectorLitModule(pl.LightningModule):
    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = build_fasterrcnn_resnet50_fpn(num_classes=num_classes, pretrained=pretrained)
        self.lr = lr
        self.weight_decay = weight_decay
        self._val_preds = []
        self._val_targets = []

    def on_validation_epoch_start(self) -> None:
        self._val_preds = []
        self._val_targets = []

    def training_step(
        self,
        batch: Tuple[List[torch.Tensor], List[Dict[str, Any]]],
        batch_idx: int,
    ) -> torch.Tensor:
        images, targets = batch
        loss_dict = self.model(images, targets)
        loss = sum(loss_dict.values())

        bs = len(images)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=bs)
        for k, v in loss_dict.items():
            self.log(f"train/{k}", v, on_step=True, on_epoch=True, batch_size=bs)

        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch

        preds = self.model(images)

        thr = 0.5
        passed = 0
        top_scores = []

        for p in preds:
            if len(p["scores"]) > 0:
                top_scores.append(float(p["scores"][0]))
                passed += int((p["scores"] >= thr).sum())

        bs = len(images)
        self.log("val/top_score_mean", float(sum(top_scores) / max(1, len(top_scores))), on_epoch=True, batch_size=bs)
        self.log("val/dets_ge_0_5", float(passed), on_epoch=True, batch_size=bs)

        preds_cpu = []
        for p in preds:
            preds_cpu.append(
                {
                    "boxes": p["boxes"].detach().cpu(),
                    "labels": p["labels"].detach().cpu(),
                    "scores": p["scores"].detach().cpu(),
                }
            )

        targets_cpu = []
        for t, img in zip(targets, images):
            targets_cpu.append(
                {
                    "boxes": t["boxes"].detach().cpu(),
                    "labels": t["labels"].detach().cpu(),
                    "image_id": t["image_id"].detach().cpu() if hasattr(t["image_id"], "detach") else t["image_id"],
                    "image_size": (int(img.shape[-2]), int(img.shape[-1])),
                }
            )

        self._val_preds.extend(preds_cpu)
        self._val_targets.extend(targets_cpu)

        num_det = sum(len(p["boxes"]) for p in preds_cpu)
        self.log("val/num_detections", float(num_det), on_epoch=True, prog_bar=True, batch_size=bs)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    def on_validation_epoch_end(self) -> None:
        metrics = coco_map_from_predictions(
            preds=self._val_preds,
            targets=self._val_targets,
            num_classes=self.hparams.num_classes - 1,  # у нас num_classes=13 (включая фон)
        )
        self.log("val/mAP", metrics["mAP"], prog_bar=True)
        self.log("val/mAP50", metrics["mAP50"], prog_bar=True)