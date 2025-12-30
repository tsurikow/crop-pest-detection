from __future__ import annotations

from pathlib import Path


def write_triton_config_pbtxt(
    *,
    out_path: str | Path,
    model_name: str,
    max_batch_size: int,
    input_h: int,
    input_w: int,
    max_dets: int,
    labels_dtype: str = "TYPE_INT64",
    instance_kind: str = "KIND_GPU",
    instance_count: int = 1,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pbtxt = f'''name: "{model_name}"
platform: "onnxruntime_onnx"
max_batch_size: {max_batch_size}

input [
  {{
    name: "image"
    data_type: TYPE_FP32
    dims: [ 3, {input_h}, {input_w} ]
  }}
]

output [
  {{
    name: "boxes"
    data_type: TYPE_FP32
    dims: [ {max_dets}, 4 ]
  }},
  {{
    name: "scores"
    data_type: TYPE_FP32
    dims: [ {max_dets} ]
  }},
  {{
    name: "labels"
    data_type: {labels_dtype}
    dims: [ {max_dets} ]
  }},
  {{
    name: "num"
    data_type: TYPE_INT64
    dims: [ 1 ]
  }}
]

instance_group [
  {{
    kind: {instance_kind}
    count: {instance_count}
  }}
]
'''
    out_path.write_text(pbtxt, encoding="utf-8")
    return out_path
