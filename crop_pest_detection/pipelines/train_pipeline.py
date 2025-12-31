from __future__ import annotations

import datetime
import subprocess
from pathlib import Path

import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from crop_pest_detection.data.datamodule import PestDataModule
from crop_pest_detection.export.export_onnx import export_onnx_from_ckpt
from crop_pest_detection.models.lit_module import PestDetectorLitModule


def _git(cmd: list[str], cwd: Path) -> str:
    return subprocess.check_output(["git", *cmd], cwd=str(cwd)).decode().strip()


def get_git_info(repo_root: Path) -> dict[str, str]:
    info: dict[str, str] = {}
    try:
        info["git.commit"] = _git(["rev-parse", "HEAD"], repo_root)
        info["git.branch"] = _git(["rev-parse", "--abbrev-ref", "HEAD"], repo_root)
        info["git.is_dirty"] = (
            "true" if _git(["status", "--porcelain"], repo_root) else "false"
        )
    except Exception:
        pass
    return info


def _find_repo_root(start: Path) -> Path:
    p = start.resolve()
    for _ in range(10):
        if (p / "configs").is_dir():
            return p
        if p.parent == p:
            break
        p = p.parent
    raise RuntimeError("Cannot find repo root (no ./configs directory found)")


def run_train(cfg: DictConfig) -> None:
    repo_root = _find_repo_root(Path.cwd())
    git_info = get_git_info(repo_root)

    pl.seed_everything(int(cfg.seed), workers=True)

    cfg_path = repo_root / "hydra_config.yaml"
    cfg_path.write_text(OmegaConf.to_yaml(cfg), encoding="utf-8")

    datamodule = PestDataModule(
        data_root=str((repo_root / cfg.paths.data_root).resolve()),
        num_classes=int(cfg.data.num_classes),
        batch_size=int(cfg.data.batch_size),
        num_workers=int(cfg.data.num_workers),
        pin_memory=bool(cfg.data.pin_memory),
    )

    model = PestDetectorLitModule(
        num_classes=int(cfg.model.num_classes),
        pretrained=bool(cfg.model.pretrained),
        lr=float(cfg.train.optim.lr),
        weight_decay=float(cfg.train.optim.weight_decay),
    )

    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(
            dirpath=str((repo_root / "checkpoints").resolve()),
            filename="epoch{epoch}-mAP50{val/mAP50:.4f}",
            monitor="val/mAP50",
            mode="max",
            save_top_k=1,
            save_last=True,
            save_weights_only=True,
        ),
    ]

    logger = True
    if bool(cfg.train.logging.use_mlflow):
        run_name = cfg.train.logging.run_name or datetime.datetime.now().strftime(
            "%Y%m%d-%H%M%S"
        )
        logger = MLFlowLogger(
            tracking_uri=str(cfg.train.logging.tracking_uri),
            experiment_name=str(cfg.train.logging.experiment_name),
            run_name=str(run_name),
        )

    trainer = pl.Trainer(
        max_epochs=int(cfg.train.trainer.max_epochs),
        accelerator=cfg.train.trainer.accelerator,
        devices=int(cfg.train.trainer.devices),
        precision=cfg.train.trainer.precision,
        log_every_n_steps=int(cfg.train.trainer.log_every_n_steps),
        limit_train_batches=cfg.train.trainer.limit_train_batches,
        limit_val_batches=cfg.train.trainer.limit_val_batches,
        num_sanity_val_steps=int(cfg.train.trainer.num_sanity_val_steps),
        deterministic=bool(cfg.train.trainer.deterministic),
        benchmark=bool(cfg.train.trainer.benchmark),
        gradient_clip_val=float(cfg.train.trainer.gradient_clip_val),
        accumulate_grad_batches=int(cfg.train.trainer.accumulate_grad_batches),
        logger=logger,
        callbacks=callbacks,
    )

    if bool(cfg.train.logging.use_mlflow) and isinstance(trainer.logger, MLFlowLogger):
        run_id = trainer.logger.run_id
        exp = trainer.logger.experiment

        trainer.logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
        exp.log_artifact(run_id, str(cfg_path))

        tags = OmegaConf.to_container(cfg.train.logging.get("tags", {}), resolve=True)
        if isinstance(tags, dict):
            for k, v in tags.items():
                exp.set_tag(run_id, str(k), str(v))

        for k, v in git_info.items():
            exp.set_tag(run_id, k, v)

    trainer.fit(model, datamodule=datamodule)

    ckpt_cb = next(
        (cb for cb in trainer.callbacks if isinstance(cb, ModelCheckpoint)), None
    )
    if not ckpt_cb:
        raise RuntimeError("ModelCheckpoint callback not found")

    export_on = str(cfg.train.export.export_on).lower()
    if export_on == "best":
        ckpt_path = ckpt_cb.best_model_path or ckpt_cb.last_model_path
    elif export_on == "last":
        ckpt_path = ckpt_cb.last_model_path or ckpt_cb.best_model_path
    else:
        raise ValueError("cfg.train.export.export_on must be one of: best, last")

    if not ckpt_path:
        raise RuntimeError("No checkpoint path found")

    onnx_path: Path | None = None
    if bool(cfg.train.export.enabled):
        out_dir = (repo_root / "exports" / "onnx").resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        onnx_path = export_onnx_from_ckpt(
            ckpt_path,
            out_dir / str(cfg.train.export.filename),
            num_classes=int(cfg.model.num_classes),
            max_dets=int(cfg.train.export.max_dets),
            score_thr=float(cfg.train.export.score_thr),
            opset=int(cfg.train.export.opset),
            h=int(cfg.train.export.input_h),
            w=int(cfg.train.export.input_w),
        )

    if isinstance(trainer.logger, MLFlowLogger):
        run_id = trainer.logger.run_id
        exp = trainer.logger.experiment

        if onnx_path is not None:
            exp.log_artifact(
                run_id,
                str(onnx_path),
                artifact_path=str(Path(cfg.train.export.artifact_path).parent),
            )
            exp.set_tag(run_id, "export.onnx", "true")

        if bool(cfg.train.export.log_ckpt_artifacts):
            if ckpt_cb.best_model_path:
                exp.log_artifact(
                    run_id, ckpt_cb.best_model_path, artifact_path="checkpoints"
                )
            if (
                ckpt_cb.last_model_path
                and ckpt_cb.last_model_path != ckpt_cb.best_model_path
            ):
                exp.log_artifact(
                    run_id, ckpt_cb.last_model_path, artifact_path="checkpoints"
                )
