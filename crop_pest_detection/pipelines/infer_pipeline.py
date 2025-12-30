from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from PIL import Image
from omegaconf import DictConfig


def _find_repo_root(start: Path) -> Path:
    p = start.resolve()
    for _ in range(15):
        if (p / "configs").is_dir():
            return p
        if p.parent == p:
            break
        p = p.parent
    raise RuntimeError("Cannot find repo root (no ./configs directory found)")


def _load_image(path: Path, h: int, w: int) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((w, h))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))  # CHW
    return arr


def _postprocess(
    cfg: DictConfig, outputs: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
) -> Dict[str, Any]:
    import torch
    from torchvision.ops import nms

    boxes, scores, labels, num = outputs

    n_raw = int(np.asarray(num).reshape(-1)[0]) if num is not None else int(len(scores))
    n = max(
        0, min(n_raw, int(boxes.shape[0]), int(scores.shape[0]), int(labels.shape[0]))
    )

    boxes_t = torch.as_tensor(boxes[:n], dtype=torch.float32)
    scores_t = torch.as_tensor(scores[:n], dtype=torch.float32)
    labels_t = torch.as_tensor(labels[:n], dtype=torch.int64)

    score_thr = float(getattr(cfg.infer.post, "score_thr", 0.0))
    nms_iou_thr = float(getattr(cfg.infer.post, "nms_iou_thr", 0.5))
    top_k = int(getattr(cfg.infer.post, "top_k", 0))
    per_class_nms = bool(getattr(cfg.infer.post, "per_class_nms", True))

    if boxes_t.numel() == 0:
        return {"num": 0, "boxes": [], "scores": [], "labels": []}

    keep = scores_t >= score_thr
    boxes_t, scores_t, labels_t = boxes_t[keep], scores_t[keep], labels_t[keep]

    if boxes_t.numel() == 0:
        return {"num": 0, "boxes": [], "scores": [], "labels": []}

    if per_class_nms:
        kept_idx = []
        for cls in torch.unique(labels_t):
            cls_idx = torch.nonzero(labels_t == cls, as_tuple=False).squeeze(1)
            cls_keep = nms(boxes_t[cls_idx], scores_t[cls_idx], nms_iou_thr)
            kept_idx.append(cls_idx[cls_keep])
        kept_idx_t = (
            torch.cat(kept_idx)
            if kept_idx
            else boxes_t.new_zeros((0,), dtype=torch.long)
        )
    else:
        kept_idx_t = nms(boxes_t, scores_t, nms_iou_thr)

    boxes_t, scores_t, labels_t = (
        boxes_t[kept_idx_t],
        scores_t[kept_idx_t],
        labels_t[kept_idx_t],
    )

    if top_k and boxes_t.shape[0] > top_k:
        order = torch.argsort(scores_t, descending=True)[:top_k]
        boxes_t, scores_t, labels_t = boxes_t[order], scores_t[order], labels_t[order]

    n_out = int(boxes_t.shape[0])
    return {
        "num": n_out,
        "boxes": boxes_t.cpu().numpy().tolist(),
        "scores": scores_t.cpu().numpy().tolist(),
        "labels": labels_t.cpu().numpy().tolist(),
    }


def _infer_onnxruntime(
    cfg: DictConfig, onnx_path: Path, image_chw: np.ndarray
) -> Dict[str, Any]:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp = image_chw.astype(np.float32)
    boxes, scores, labels, num = sess.run(
        ["boxes", "scores", "labels", "num"], {"image": inp}
    )
    return _postprocess(cfg, (boxes, scores, labels, num))


def _infer_triton_http(cfg: DictConfig, image_chw: np.ndarray) -> Dict[str, Any]:
    import tritonclient.http as httpclient
    from tritonclient.utils import np_to_triton_dtype

    url = str(cfg.infer.triton.url)
    model_name = str(cfg.infer.triton.model_name)

    client = httpclient.InferenceServerClient(url=url, verbose=False)

    inp = image_chw.astype(np.float32)
    inputs = [httpclient.InferInput("image", inp.shape, np_to_triton_dtype(inp.dtype))]
    inputs[0].set_data_from_numpy(inp)

    outputs = [
        httpclient.InferRequestedOutput("boxes"),
        httpclient.InferRequestedOutput("scores"),
        httpclient.InferRequestedOutput("labels"),
        httpclient.InferRequestedOutput("num"),
    ]

    res = client.infer(model_name=model_name, inputs=inputs, outputs=outputs)
    boxes = res.as_numpy("boxes")
    scores = res.as_numpy("scores")
    labels = res.as_numpy("labels")
    num = res.as_numpy("num")
    return _postprocess(cfg, (boxes, scores, labels, num))


def run_infer(cfg: DictConfig) -> Path:
    repo_root = _find_repo_root(Path.cwd())

    in_path = (repo_root / str(cfg.infer.input_path)).resolve()
    out_path = (repo_root / str(cfg.infer.output_path)).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    h = int(cfg.infer.export.input_h)
    w = int(cfg.infer.export.input_w)

    image = _load_image(in_path, h=h, w=w)

    backend = str(cfg.infer.backend).lower()
    if backend == "onnxruntime":
        onnx_path = (repo_root / str(cfg.infer.onnx_path)).resolve()
        pred = _infer_onnxruntime(cfg, onnx_path, image)
    elif backend == "triton_http":
        pred = _infer_triton_http(cfg, image)
    else:
        raise ValueError("infer.backend must be one of: onnxruntime, triton_http")

    payload = {"input": str(in_path), "backend": backend, "pred": pred}
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return out_path
