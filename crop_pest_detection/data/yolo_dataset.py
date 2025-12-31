from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as F


@dataclass(frozen=True)
class YoloDatasetPaths:
    images_dir: Path
    labels_dir: Path


def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """PIL RGB -> float32 tensor [C,H,W] in [0,1]."""
    return F.to_tensor(image)


class YoloPestDetectionDataset(Dataset):
    """
    Dataset for object detection with YOLO txt labels.

    Directory structure:
      root/
        train|valid|test/
          images/*.jpg
          labels/*.txt

    Label format per line:
      class_id cx cy w h   (all normalized to [0, 1])
    """

    def __init__(
        self,
        root_dir: str | Path,
        split: str,
        num_classes: int = 12,
        image_dir_name: str = "images",
        labels_dir_name: str = "labels",
        transforms=None,
        strict: bool = True,
    ) -> None:
        super().__init__()
        self.root_dir = Path(root_dir)
        self.split = split
        self.num_classes = num_classes
        self.transforms = transforms
        self.strict = strict

        split_dir = self.root_dir / split
        self.paths = YoloDatasetPaths(
            images_dir=split_dir / image_dir_name,
            labels_dir=split_dir / labels_dir_name,
        )

        if not self.paths.images_dir.is_dir():
            raise RuntimeError(f"Images directory not found: {self.paths.images_dir}")
        if not self.paths.labels_dir.is_dir():
            raise RuntimeError(f"Labels directory not found: {self.paths.labels_dir}")

        exts = {".jpg", ".jpeg", ".png"}
        self.image_paths: List[Path] = sorted(
            p for p in self.paths.images_dir.iterdir() if p.suffix.lower() in exts
        )
        if not self.image_paths:
            raise RuntimeError(f"No images found in {self.paths.images_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def _read_yolo_labels(
        self, label_path: Path, width: int, height: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        boxes: List[List[float]] = []
        labels: List[int] = []

        if not label_path.exists():
            if self.strict:
                raise RuntimeError(f"Missing label file: {label_path}")
            return torch.zeros((0, 4), dtype=torch.float32), torch.zeros(
                (0,), dtype=torch.int64
            )

        with label_path.open("r") as f:
            for ln, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    if self.strict:
                        raise RuntimeError(
                            f"Bad label line (len!=5) in {label_path}:{ln}: {line}"
                        )
                    continue

                class_id = int(parts[0])
                if not (0 <= class_id < self.num_classes):
                    msg = f"class_id out of range [0,{self.num_classes - 1}] in {label_path}:{ln}: {class_id}"
                    if self.strict:
                        raise RuntimeError(msg)
                    continue

                cx = float(parts[1]) * width
                cy = float(parts[2]) * height
                bw = float(parts[3]) * width
                bh = float(parts[4]) * height

                x_min = cx - bw / 2.0
                y_min = cy - bh / 2.0
                x_max = cx + bw / 2.0
                y_max = cy + bh / 2.0

                x_min = max(0.0, min(x_min, width - 1.0))
                x_max = max(0.0, min(x_max, width - 1.0))
                y_min = max(0.0, min(y_min, height - 1.0))
                y_max = max(0.0, min(y_max, height - 1.0))

                labels.append(class_id + 1)
                boxes.append([x_min, y_min, x_max, y_max])

        if boxes:
            return torch.tensor(boxes, dtype=torch.float32), torch.tensor(
                labels, dtype=torch.int64
            )

        return torch.zeros((0, 4), dtype=torch.float32), torch.zeros(
            (0,), dtype=torch.int64
        )

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        img_path = self.image_paths[idx]
        label_path = self.paths.labels_dir / f"{img_path.stem}.txt"

        image_pil = Image.open(img_path).convert("RGB")
        width, height = image_pil.size

        boxes, labels = self._read_yolo_labels(label_path, width, height)

        target: Dict[str, Any] = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
        }

        if self.transforms is not None:
            out = (
                self.transforms(image_pil, target)
                if callable(self.transforms)
                else self.transforms(image_pil)
            )
            if isinstance(out, tuple) and len(out) == 2:
                image_pil, target = out
            else:
                image_pil = out

        if isinstance(image_pil, Image.Image):
            image = _pil_to_tensor(image_pil)
        else:
            image = image_pil

        return image, target


def detection_collate_fn(
    batch: List[Tuple[torch.Tensor, Dict[str, Any]]],
) -> Tuple[List[torch.Tensor], List[Dict[str, Any]]]:
    images, targets = list(zip(*batch))
    return list(images), list(targets)
