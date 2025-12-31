from __future__ import annotations

from pathlib import Path

import torch

from crop_pest_detection.export.wrapper import TritonDetectorWrapper
from crop_pest_detection.models.detector import build_fasterrcnn_resnet50_fpn


def _load_state(ckpt_path: str) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    return {k.replace("model.", "", 1): v for k, v in state.items()}


def load_detector_from_ckpt(ckpt_path: str, num_classes: int) -> torch.nn.Module:
    state = _load_state(ckpt_path)
    model = build_fasterrcnn_resnet50_fpn(num_classes=num_classes, pretrained=False)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def export_onnx_from_ckpt(
    ckpt_path: str,
    out_path: str | Path,
    *,
    num_classes: int,
    max_dets: int,
    score_thr: float,
    opset: int,
    h: int,
    w: int,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    detector = load_detector_from_ckpt(ckpt_path, num_classes=num_classes)
    wrapped = TritonDetectorWrapper(
        detector, max_dets=max_dets, score_thr=score_thr
    ).eval()
    dummy = torch.zeros(3, h, w, dtype=torch.float32)

    torch.onnx.export(
        wrapped,
        (dummy,),
        str(out_path),
        opset_version=int(opset),
        do_constant_folding=True,
        input_names=["image"],
        output_names=["boxes", "scores", "labels", "num"],
        dynamo=False,
    )
    return out_path
