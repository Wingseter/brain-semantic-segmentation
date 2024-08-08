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

from dataModule import *
from modelModule import *
from src.utils import *

@hydra.main(version_base=None, config_path="conf", config_name="config")
def train(cfg:DictConfig) -> None:
    # Create Unique run id by date and time
    run_id = generate_run_id()

    L.seed_everything(cfg.seed, workers=True)
    torch.set_float32_matmul_precision("high")

    # initialize DataModules
    ## TODO this is temperary you have to implement datamodule
    ## TODO You have to make it changable with cfg files 
    dm = BrainDataModule(
        root
    )
    dm.setup()

    model = ModelModule(
        model_select = cfg.model_name,
        leanring_rate = cfg.learning_rate,
        use_scheduler = cfg.use_scheduler,
    )

    logger = TensorBoardLogger(save_dir=cfg.logs_dir, name = "", version=run_id)

    checkpoint_callback = ModelCheckpoint(
        monitor = "val_loss",
        mode = "min",
        save_top_k = 2,
        dirpath = os.path.join(cfg.checkpoint_dirpath, run_id),
        filename="{epoch}-{step}-{val_loss:.2f}-{val_dice:2f}",
    )

    lr_monitor = LearningRateMonitor(loggin_interval = "step")

    ## TODO You have to make it changable with cfg files 
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=cfg.early_stopping_patient,
        verbose=True,
        mode="min",
    )

    trainer = L.trainer(
        max_epochs=cfg.max_epochs,
        accelerator="auto",
        devices="auto",
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor, early_stopping],
    )

    trainer.fit(model.dm)

if __name__ == "main":
    train()


