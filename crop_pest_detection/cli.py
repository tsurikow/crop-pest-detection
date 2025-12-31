from __future__ import annotations

from pathlib import Path
from typing import List

import fire
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf


def _find_repo_root(start: Path) -> Path:
    p = start.resolve()
    for _ in range(15):
        if (p / "configs").is_dir():
            return p
        if p.parent == p:
            break
        p = p.parent
    raise RuntimeError("Cannot find repo root (no ./configs directory found)")


def compose_cfg(
    config_name: str, overrides: List[str] | None = None, repo_root: Path | None = None
) -> DictConfig:
    overrides = list(overrides or [])

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    repo_root = repo_root or _find_repo_root(Path.cwd())
    config_dir = (repo_root / "configs").resolve()

    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name=config_name, overrides=overrides)

    return cfg


class CLI:
    def cfg(self, config_name: str, *overrides: str) -> None:
        cfg = compose_cfg(config_name, list(overrides))
        print(OmegaConf.to_yaml(cfg))

    def train(self, *overrides: str) -> None:
        cfg = compose_cfg("train", list(overrides))
        from crop_pest_detection.pipelines.train_pipeline import run_train

        run_train(cfg)

    def export_onnx(self, *overrides: str) -> None:
        cfg = compose_cfg("infer", list(overrides))
        from crop_pest_detection.pipelines.export_pipeline import run_export_onnx

        run_export_onnx(cfg)

    def triton_build_repo(self, *overrides: str) -> None:
        cfg = compose_cfg("infer", list(overrides))
        from crop_pest_detection.pipelines.triton_pipeline import run_triton_build_repo

        run_triton_build_repo(cfg)

    def infer(self, *overrides: str) -> None:
        cfg = compose_cfg("infer", list(overrides))
        from crop_pest_detection.pipelines.infer_pipeline import run_infer

        run_infer(cfg)

    def visualize(self, *overrides: str):
        cfg = compose_cfg(config_name="viz", overrides=list(overrides))
        from crop_pest_detection.pipelines.visualize_pipeline import run_visualize

        return str(run_visualize(cfg))


def main() -> None:
    fire.Fire(CLI)


if __name__ == "__main__":
    main()
