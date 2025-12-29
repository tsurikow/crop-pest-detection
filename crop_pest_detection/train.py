from pathlib import Path
import datetime

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import MLFlowLogger

from crop_pest_detection.data.datamodule import PestDataModule
from crop_pest_detection.models.lit_module import PestDetectorLitModule


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.seed, workers=True)

    cfg_path = Path("hydra_config.yaml")
    cfg_path.write_text(OmegaConf.to_yaml(cfg), encoding="utf-8")

    datamodule = PestDataModule(
        data_root=cfg.paths.data_root,
        num_classes=cfg.data.num_classes,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )

    model = PestDetectorLitModule(
        num_classes=cfg.model.num_classes,
        pretrained=cfg.model.pretrained,
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
    )

    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(
            dirpath="checkpoints",
            filename="epoch{epoch}-mAP50{val/mAP50:.4f}",
            monitor="val/mAP50",
            mode="max",
            save_top_k=1,
            save_last=True,
            save_weights_only=True,
        )
    ]

    logger = True
    if cfg.logging.use_mlflow:
        run_name = cfg.logging.run_name or datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        logger = MLFlowLogger(
            tracking_uri=cfg.logging.tracking_uri,
            experiment_name=cfg.logging.experiment_name,
            run_name=run_name,
        )

    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        limit_train_batches=cfg.trainer.limit_train_batches,
        limit_val_batches=cfg.trainer.limit_val_batches,
        num_sanity_val_steps=cfg.trainer.num_sanity_val_steps,
        logger=logger,
        callbacks=callbacks,
    )

    if cfg.logging.use_mlflow and isinstance(trainer.logger, MLFlowLogger):
        trainer.logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
        trainer.logger.experiment.log_artifact(trainer.logger.run_id, str(cfg_path))

    trainer.fit(model, datamodule=datamodule)

    if cfg.logging.use_mlflow and isinstance(trainer.logger, MLFlowLogger):
        ckpt_cb = next((cb for cb in trainer.callbacks if isinstance(cb, ModelCheckpoint)), None)
        if ckpt_cb and ckpt_cb.best_model_path:
            trainer.logger.experiment.log_artifact(trainer.logger.run_id, ckpt_cb.best_model_path)
        if ckpt_cb and ckpt_cb.last_model_path:
            trainer.logger.experiment.log_artifact(trainer.logger.run_id, ckpt_cb.last_model_path)


if __name__ == "__main__":
    main()