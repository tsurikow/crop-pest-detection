from __future__ import annotations

from pathlib import Path

from mlflow.tracking import MlflowClient


def download_artifact(
    tracking_uri: str, run_id: str, artifact_path: str, dst_dir: str | Path
) -> Path:
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    client = MlflowClient(tracking_uri=tracking_uri)
    local_path = client.download_artifacts(run_id, artifact_path, dst_path=str(dst_dir))
    return Path(local_path).resolve()
