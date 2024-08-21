import os

import hydra
import torch

import lightning as L
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig

from src.dataModule import *
from src.modelModule import *
from src.utils import *

@hydra.main(version_base=None, config_path="conf", config_name="config")
def train(cfg:DictConfig) -> None:
    # Create Unique run id by date and time
    run_id = generate_run_id()

    seed_everything(cfg.seed)
    L.seed_everything(cfg.seed, workers=True)

    torch.set_float32_matmul_precision("high")

    # initialize DataModules
    dm = DataModule(
        train_batch_size=cfg.experiment.train_batch_size,
        valid_batch_size=cfg.experiment.valid_batch_size,
        num_workers=cfg.num_workers,
        transform_name=cfg.transform_name,
        dataset_name=cfg.dataset_name,
        dataset_params=cfg.dataset_params
    )

    model = ModelModule(
        model_name = cfg.model_name,
        learning_rate = cfg.learning_rate,
        use_scheduler = cfg.use_scheduler,
        model_params = cfg.model_params
        epochs= cfg.max_epochs 
    )

    logger = TensorBoardLogger(save_dir=cfg.logs_dir, name = "", version=run_id)

    checkpoint_callback = ModelCheckpoint(
        monitor = "val_loss",
        mode = "min",
        save_top_k = 2,
        dirpath = os.path.join(cfg.checkpoint_dirpath, run_id),
        filename="{epoch}-{step}-{val_loss:.2f}-{val_dice:2f}",
    )

    lr_monitor = LearningRateMonitor(logging_interval = "step")

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=cfg.experiment.early_stopping_patient,
        verbose=True,
        mode="min",
    )

    trainer = L.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator=cfg.device,
        devices="auto",
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor, early_stopping],
        log_every_n_steps=30
    )

    ckpt_path = cfg.get('ckpt_path', None)
    if ckpt_path and os.path.exists(ckpt_path):
        print(f"Resuming from checkpoint: {ckpt_path}")
    else:
        ckpt_path = None

    # Train the model
    trainer.fit(model, dm, ckpt_path=ckpt_path)

if __name__ == "__main__":
    train()

