from __future__ import annotations

from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .yolo_dataset import YoloPestDetectionDataset, detection_collate_fn


class PestDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root: str,
        num_classes: int = 12,
        batch_size: int = 2,
        num_workers: int = 4,
        pin_memory: bool = True,
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            self.train_ds = YoloPestDetectionDataset(
                root_dir=self.data_root,
                split="train",
                num_classes=self.num_classes,
                transforms=None,
                strict=True,
            )
            self.val_ds = YoloPestDetectionDataset(
                root_dir=self.data_root,
                split="valid",
                num_classes=self.num_classes,
                transforms=None,
                strict=True,
            )

        if stage in (None, "test"):
            self.test_ds = YoloPestDetectionDataset(
                root_dir=self.data_root,
                split="test",
                num_classes=self.num_classes,
                transforms=None,
                strict=True,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=detection_collate_fn,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=detection_collate_fn,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=detection_collate_fn,
            pin_memory=self.pin_memory,
        )
