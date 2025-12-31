from __future__ import annotations

import shutil
from pathlib import Path

from omegaconf import DictConfig

from crop_pest_detection.triton.config_writer import write_triton_config_pbtxt


def _find_repo_root(start: Path) -> Path:
    p = start.resolve()
    for _ in range(15):
        if (p / "configs").is_dir():
            return p
        if p.parent == p:
            break
        p = p.parent
    raise RuntimeError("Cannot find repo root (no ./configs directory found)")


def run_triton_build_repo(cfg: DictConfig) -> Path:
    repo_root = _find_repo_root(Path.cwd())

    onnx_path = (repo_root / str(cfg.infer.onnx_path)).resolve()
    model_repo = (repo_root / str(cfg.infer.triton.model_repository)).resolve()
    model_name = str(cfg.infer.triton.model_name)
    model_version = str(cfg.infer.triton.model_version)

    input_h = int(cfg.infer.export.input_h)
    input_w = int(cfg.infer.export.input_w)
    max_dets = int(cfg.infer.export.max_dets)

    labels_dtype = str(cfg.infer.triton.get("labels_dtype", "TYPE_INT64"))
    instance_kind = str(cfg.infer.triton.get("instance_kind", "KIND_GPU"))
    instance_count = int(cfg.infer.triton.get("instance_count", 1))

    model_dir = model_repo / model_name
    version_dir = model_dir / model_version
    version_dir.mkdir(parents=True, exist_ok=True)

    dst_onnx = version_dir / "model.onnx"
    shutil.copy2(onnx_path, dst_onnx)

    cfg_pbtxt = model_dir / "config.pbtxt"
    write_triton_config_pbtxt(
        out_path=cfg_pbtxt,
        model_name=model_name,
        max_batch_size=0,
        input_h=input_h,
        input_w=input_w,
        max_dets=max_dets,
        labels_dtype=labels_dtype,
        instance_kind=instance_kind,
        instance_count=instance_count,
    )

    return model_repo
