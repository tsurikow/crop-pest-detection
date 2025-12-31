# Crop Pest Detection

Automatic detection of agricultural pests in images using **Faster R-CNN ResNet50-FPN** and an MLOps stack: **uv**, **DVC**, **MLflow**, **Triton Inference Server**, **MinIO**, **PostgreSQL**, **docker-compose**.

## 1) Quick start

### Requirements

- Python **3.12**
- **uv**
- **Docker** + Docker Compose
- For GPU inference/training on Linux/WSL2: NVIDIA Driver + NVIDIA Container Toolkit

> On macOS (MPS), training is possible, but some `torchvision` detection ops can be unstable. For stability, use CPU or a Linux GPU.

### Install dependencies

```bash
# create venv and install dependencies
uv sync

# (optional) enable pre-commit
uv run pre-commit install
uv run pre-commit run -a
```

## 2) Data (DVC)

The dataset is stored **outside git** and managed with **DVC**.

### Download the dataset

```bash
# pull data from the DVC remote
uv run dvc pull
```

After `dvc pull`, data folders will appear in the project (e.g., `raw_data/...`).

> If you keep a manual copy of the dataset in `raw_data/`, make sure DVC is configured correctly and the path matches `configs/paths/default.yaml`.

## 3) MLflow + MinIO + Postgres (docker-compose)

### Environment variables (.env)

`docker compose` reads environment variables from the `.env` file in the repository root.

- The repository contains an example: `.env.example`
- For local runs, create your own `.env`:

```bash
cp .env.example .env
```

### Start the infrastructure

**Running without a profile** starts only the MLOps stack (Postgres + MinIO + MLflow) — without Triton.

```bash
docker compose up -d
```

Checks:

- MLflow UI: http://localhost:8080
- MinIO Console: http://localhost:9001

> If you don’t want to start everything, you can start specific services only (e.g., the MLflow stack without Triton).

## 4) Training

Training is launched via the `cpd` CLI (Fire + Hydra compose API). Configs live in `configs/`.

### Sanity check (short run)

```bash
uv run cpd train \
  train.trainer.max_epochs=1 \
  train.trainer.limit_train_batches=0.02 \
  train.trainer.limit_val_batches=0.02
```

### Full GPU training (example)

```bash
uv run cpd train \
  train.trainer.accelerator=gpu train.trainer.devices=1 train.trainer.precision=16-mixed \
  train.trainer.max_epochs=25 \
  data.batch_size=4 data.num_workers=8 data.pin_memory=true \
  train.trainer.log_every_n_steps=50 \
  train.trainer.limit_train_batches=1.0 train.trainer.limit_val_batches=1.0
```

Outputs:

- checkpoints: `checkpoints/` (git-ignored)
- exports: `exports/` (git-ignored)
- logs in MLflow (metrics/params/artifacts)

## 5) Export the model to ONNX

Export can be done automatically after training (see `train.export.*`) or manually via a dedicated pipeline.

### Export from a local checkpoint

```bash
uv run cpd export_onnx \
  infer.source=local \
  infer.ckpt_path=checkpoints/last.ckpt \
  infer.onnx_path=exports/onnx/model.onnx \
  infer.export.opset=17 \
  infer.export.input_h=640 infer.export.input_w=640 \
  infer.export.score_thr=0.05
```

### Export/download from MLflow

Supported modes:

- `infer.source=mlflow_ckpt` — download a checkpoint from MLflow and export
- `infer.source=mlflow_onnx` — download an already exported ONNX from MLflow

Example:

```bash
uv run cpd export_onnx \
  infer.source=mlflow_onnx \
  infer.mlflow.tracking_uri=http://localhost:8080 \
  infer.mlflow.run_id=<RUN_ID> \
  infer.mlflow.onnx_artifact_path=onnx/model.onnx \
  infer.onnx_path=exports/onnx/model.onnx
```

## 6) Triton: build a model_repository and run the server

### Build the model_repository

```bash
uv run cpd triton_build_repo \
  infer.onnx_path=exports/onnx/model.onnx \
  infer.triton.model_repository=model_repository \
  infer.triton.model_name=crop_pest_detector \
  infer.triton.model_version=1 \
  infer.export.input_h=640 infer.export.input_w=640 \
  infer.export.max_dets=100
```

By default, `config.pbtxt` is generated for GPU (`instance_kind=KIND_GPU`). For CPU, override: `infer.triton.instance_kind=KIND_CPU`.

### Run Triton (without stopping other containers)

Triton is started **via profiles**:

- **GPU**: profile `gpu` (service `triton`)
- **CPU**: profile `cpu` (service `triton-cpu`)

```bash
# GPU Triton
docker compose --profile gpu up -d triton

# or CPU Triton
docker compose --profile cpu up -d triton-cpu

# logs (GPU Triton)
docker compose logs -f triton
```

Health checks:

- HTTP: http://localhost:8000/v2/health/ready
- Metrics: http://localhost:8002/metrics

## 7) Inference

### Inference via Triton HTTP

```bash
uv run cpd infer \
  infer.backend=triton_http \
  infer.triton.url=localhost:8000 \
  infer.triton.model_name=crop_pest_detector \
  infer.input_path=raw_data/agro_pest/valid/images/<IMAGE>.jpg \
  infer.output_path=outputs/infer/result.json
```

### Inference via onnxruntime (CPU)

```bash
uv run cpd infer \
  infer.backend=onnxruntime \
  infer.onnx_path=exports/onnx/model.onnx \
  infer.input_path=raw_data/agro_pest/valid/images/<IMAGE>.jpg \
  infer.output_path=outputs/infer/result.json
```

## 8) Prediction visualization

Class names are read from `raw_data/agro_pest/data.yaml`.

```bash
uv run cpd visualize \
  viz.input_json=outputs/infer/result.json \
  viz.output_path=outputs/infer/result.png \
  viz.score_thr=0.30 \
  viz.yolo_data_yaml=raw_data/agro_pest/data.yaml
```

### Demo (inference + visualization)

Example inference output on a single image (score_thr=0.50):

![Inference demo](assets/infer_visual_example.png)

## 9) Repository layout (short)

- `crop_pest_detection/` — package code
- `configs/` — Hydra configs (train/infer/paths/model/data)
- `scripts/` — helper scripts (if any)
- `model_repository/` — Triton model repository (git keeps only the config; weights/onnx are ignored)
- `raw_data/`, `checkpoints/`, `exports/`, `outputs/`, `downloads/` — data/artifacts (git-ignored)

## 10) Common issues

### Triton won’t start with GPU

If you run the `gpu` profile and see an error like `could not select device driver ... with capabilities: [[gpu]]`, Docker cannot see an NVIDIA GPU.

Check:

- NVIDIA Driver is installed
- NVIDIA Container Toolkit is installed
- Docker is running in an environment where the GPU is visible (on WSL2, Docker Desktop must be running)

### /models mount error on WSL2

If you see `error mounting ... to rootfs at "/models": ... no such file or directory`, it’s usually a Docker Desktop/WSL bind-mount path issue. Make sure the `model_repository` path exists inside WSL and is accessible to Docker.

---

## Commands (cheat sheet)

```bash
# infrastructure
docker compose up -d  # MLOps stack (no profile)

# training
uv run cpd train ...

# export
uv run cpd export_onnx ...

# triton repo + run triton
uv run cpd triton_build_repo ...
docker compose --profile gpu up -d triton      # Triton GPU
docker compose --profile cpu up -d triton-cpu  # Triton CPU

# inference + visualization
uv run cpd infer ...
uv run cpd visualize ...
```
