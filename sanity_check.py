#!/usr/bin/env python
"""Sanity-check training: mini dataset (seq 00), batch_size=1, 50 steps + 1 val.

Runs with CUDA_LAUNCH_BLOCKING=1 for synchronous error reporting.
Usage:
    CUDA_VISIBLE_DEVICES=0 python sanity_check.py
"""
import os
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

import misc as misc
import torch
from voxdet_core import Config
from voxdet_models import *  # noqa: F401,F403  — register modules
import pytorch_lightning as pl
from LightningTools.pl_model import pl_model
from LightningTools.dataset_dm import DataModule
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor


def main():
    config_path = "configs/voxdet-semantickitti-cam.py"
    log_folder = "logs/sanity_check"
    misc.check_path(log_folder)

    cfg = Config.fromfile(config_path)

    # --- Overrides for quick sanity check ---
    # Use mini_split: train on seq 00 only, val on seq 08
    cfg.data.train.mini_split = True
    cfg.data.val.mini_split = True
    cfg.data.test.mini_split = True

    # Small batch, fewer workers
    cfg.train_dataloader_config.batch_size = 1
    cfg.train_dataloader_config.num_workers = 2
    cfg.test_dataloader_config.batch_size = 1
    cfg.test_dataloader_config.num_workers = 2

    # Very short training: 50 steps, validate after
    cfg.training_steps = 50
    cfg.lr_scheduler.total_steps = 60

    # Inject CLI-expected keys
    cfg.log_folder = log_folder
    cfg.ckpt_path = None
    cfg.resume = False
    cfg.seed = 7240
    cfg.save_path = None
    cfg.test_mapping = False
    cfg.submit = False
    cfg.eval = False
    cfg.log_every_n_steps = 10
    cfg.check_val_every_n_epoch = 1
    cfg.pretrain = False

    pl.seed_everything(cfg.seed)

    model = pl_model(cfg)
    data_dm = DataModule(cfg)
    data_dm.setup()

    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=log_folder, name="tensorboard"
    )

    trainer = pl.Trainer(
        devices=1,
        accelerator="gpu",
        max_steps=cfg.training_steps,
        callbacks=[LearningRateMonitor(logging_interval="step")],
        logger=[tb_logger],
        log_every_n_steps=cfg.log_every_n_steps,
        check_val_every_n_epoch=cfg.check_val_every_n_epoch,
        enable_checkpointing=False,
        num_sanity_val_steps=0,
    )

    print(f"=== Sanity check: {cfg.training_steps} train steps, "
          f"batch_size=1, mini_split (seq 00) ===")
    print(f"CUDA_LAUNCH_BLOCKING={os.environ.get('CUDA_LAUNCH_BLOCKING', 'unset')}")
    trainer.fit(model=model, datamodule=data_dm)
    print("=== Sanity check PASSED ===")


if __name__ == "__main__":
    main()
