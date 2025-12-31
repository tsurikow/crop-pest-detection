from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from omegaconf import DictConfig

from crop_pest_detection.visualize import draw_detections


def _find_repo_root(start: Path) -> Path:
    p = start.resolve()
    for _ in range(20):
        if (p / "configs").is_dir():
            return p
        if p.parent == p:
            break
        p = p.parent
    raise RuntimeError("Cannot find repo root (no ./configs directory found)")


def run_visualize(cfg: DictConfig) -> Path:
    repo_root = _find_repo_root(Path.cwd())

    in_json = (repo_root / str(cfg.viz.input_json)).resolve()
    out_path = (repo_root / str(cfg.viz.output_path)).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data: Dict[str, Any] = json.loads(in_json.read_text(encoding="utf-8"))
    image_path = data["input"]
    pred = data["pred"]

    yolo_yaml = cfg.viz.get("yolo_data_yaml", None)
    yolo_yaml = str((repo_root / str(yolo_yaml)).resolve()) if yolo_yaml else None

    return draw_detections(
        image_path=image_path,
        pred=pred,
        out_path=out_path,
        score_thr=float(cfg.viz.score_thr),
        box_format=str(cfg.viz.box_format),
        yolo_data_yaml=yolo_yaml,
    )
