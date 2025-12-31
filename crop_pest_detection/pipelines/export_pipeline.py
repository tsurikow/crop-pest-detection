from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig

from crop_pest_detection.export.export_onnx import export_onnx_from_ckpt
from crop_pest_detection.utils.mlflow_io import download_artifact


def _find_repo_root(start: Path) -> Path:
    p = start.resolve()
    for _ in range(15):
        if (p / "configs").is_dir():
            return p
        if p.parent == p:
            break
        p = p.parent
    raise RuntimeError("Cannot find repo root (no ./configs directory found)")


def run_export_onnx(cfg: DictConfig) -> Path:
    repo_root = _find_repo_root(Path.cwd())

    out_path = (repo_root / str(cfg.infer.onnx_path)).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    source = str(cfg.infer.source).lower()
    dl_dir = (repo_root / str(cfg.infer.download_dir)).resolve()
    dl_dir.mkdir(parents=True, exist_ok=True)

    if source == "mlflow_onnx":
        p = download_artifact(
            tracking_uri=str(cfg.infer.mlflow.tracking_uri),
            run_id=str(cfg.infer.mlflow.run_id),
            artifact_path=str(cfg.infer.mlflow.onnx_artifact_path),
            dst_dir=dl_dir,
        )
        out_path.write_bytes(Path(p).read_bytes())
        return out_path

    if source == "local":
        ckpt_path = (repo_root / str(cfg.infer.ckpt_path)).resolve()
    elif source == "mlflow_ckpt":
        ckpt_path = Path(
            download_artifact(
                tracking_uri=str(cfg.infer.mlflow.tracking_uri),
                run_id=str(cfg.infer.mlflow.run_id),
                artifact_path=str(cfg.infer.mlflow.ckpt_artifact_path),
                dst_dir=dl_dir,
            )
        ).resolve()
    else:
        raise ValueError("infer.source must be one of: local, mlflow_ckpt, mlflow_onnx")

    return export_onnx_from_ckpt(
        str(ckpt_path),
        out_path,
        num_classes=int(cfg.model.num_classes),
        max_dets=int(cfg.infer.export.max_dets),
        score_thr=float(cfg.infer.export.score_thr),
        opset=int(cfg.infer.export.opset),
        h=int(cfg.infer.export.input_h),
        w=int(cfg.infer.export.input_w),
    )
