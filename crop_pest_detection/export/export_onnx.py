from pathlib import Path
import torch

from crop_pest_detection.models.detector import build_fasterrcnn_resnet50_fpn
from crop_pest_detection.export.wrapper import TritonDetectorWrapper


def load_detector_from_ckpt(ckpt_path: str, num_classes: int) -> torch.nn.Module:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["state_dict"]
    state = {k.replace("model.", "", 1): v for k, v in state.items()}

    model = build_fasterrcnn_resnet50_fpn(num_classes=num_classes, pretrained=False)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def main():
    ckpt_path = "checkpoints/last.ckpt"
    out_dir = Path("exports/onnx")
    out_dir.mkdir(parents=True, exist_ok=True)

    detector = load_detector_from_ckpt(ckpt_path, num_classes=13)
    wrapped = TritonDetectorWrapper(detector, max_dets=100, score_thr=0.05).eval()

    dummy = torch.zeros(3, 640, 640, dtype=torch.float32)

    torch.onnx.export(
        wrapped,
        (dummy,),
        str(out_dir / "model.onnx"),
        opset_version=17,
        do_constant_folding=True,
        input_names=["image"],
        output_names=["boxes", "scores", "labels", "num"],
        dynamo=False,
    )


if __name__ == "__main__":
    main()