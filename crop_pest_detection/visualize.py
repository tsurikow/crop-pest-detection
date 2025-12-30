from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont


def _load_class_names_from_yolo_data_yaml(path: str | Path) -> List[str]:
    cfg = OmegaConf.load(str(path))
    names = cfg.get("names")
    if names is None:
        raise ValueError(f"No 'names' in {path}")
    return list(names)


def _to_xyxy(box: List[float], fmt: str) -> Tuple[float, float, float, float]:
    if fmt == "xywh":
        x, y, w, h = box
        return x, y, x + w, y + h
    x1, y1, x2, y2 = box
    return x1, y1, x2, y2


def _auto_label_offset(labels: List[int], num_names: int) -> int:
    # если labels в [1..num_names], а names в [0..num_names-1], делаем -1
    if not labels:
        return 0
    mn, mx = min(labels), max(labels)
    if mn >= 1 and mx <= num_names:
        return -1
    return 0


def draw_detections(
    image_path: str | Path,
    pred: Dict[str, Any],
    out_path: str | Path,
    *,
    box_format: str = "xyxy",
    score_thr: float = 0.0,
    yolo_data_yaml: Optional[str | Path] = None,  # <-- сюда data.yaml
    line_width: int = 3,
    font_size: int = 18,
) -> Path:
    image_path = Path(image_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    class_names: Optional[List[str]] = None
    if yolo_data_yaml:
        class_names = _load_class_names_from_yolo_data_yaml(yolo_data_yaml)

    boxes = pred.get("boxes", []) or []
    scores = pred.get("scores", []) or []
    labels = pred.get("labels", []) or []

    labels_int = [int(x) for x in labels]
    offset = _auto_label_offset(labels_int, len(class_names) if class_names else 0)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    for box, score, label in zip(boxes, scores, labels_int):
        score_f = float(score)
        if score_f < score_thr:
            continue

        x1, y1, x2, y2 = _to_xyxy([float(v) for v in box], box_format)

        x1 = max(0.0, min(x1, img.width - 1))
        y1 = max(0.0, min(y1, img.height - 1))
        x2 = max(0.0, min(x2, img.width - 1))
        y2 = max(0.0, min(y2, img.height - 1))

        cls_id = int(label) + offset

        if class_names and 0 <= cls_id < len(class_names):
            text = f"{class_names[cls_id]} {score_f:.2f}"
        else:
            text = f"id={label} {score_f:.2f}"

        r = (cls_id * 53) % 256
        g = (cls_id * 97) % 256
        b = (cls_id * 193) % 256
        color = (r, g, b)

        for i in range(line_width):
            draw.rectangle([x1 - i, y1 - i, x2 + i, y2 + i], outline=color)

        tw, th = draw.textbbox((0, 0), text, font=font)[2:]
        pad = 3
        tx1, ty1 = x1, max(0.0, y1 - (th + 2 * pad))
        tx2, ty2 = x1 + tw + 2 * pad, ty1 + th + 2 * pad
        draw.rectangle([tx1, ty1, tx2, ty2], fill=color)
        draw.text((tx1 + pad, ty1 + pad), text, fill=(255, 255, 255), font=font)

    img.save(out_path)
    return out_path
