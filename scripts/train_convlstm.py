#!/usr/bin/env python

import os
import time

import torch

from torch.utils.data import DataLoader

from datetime import datetime

from utils.config import *
from utils.logger import LoggerFactory

from utils.data_loader import ERA5DataLoader
from utils.feature_builder import FeatureBuilder

from datasets.convlstm_dataset import ConvLSTMDataset

from models.forecast_convlstm import ForecastConvLSTM

from training.trainer import Trainer


STAMP = datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)

MODEL_TYPE = "convlstm"
BATCH_SIZE = 32
NUM_WORKERS = 8

LOG_FILE = (
    f"logs/train_{MODEL_TYPE}_{STAMP}.log"
)


def main():

    logger = LoggerFactory.create(
        LOG_FILE
    )

    logger.info(
        "Starting ConvLSTM training run"
    )

    ########################################################
    # Load ERA5
    ########################################################

    loader = ERA5DataLoader(
        DATASET
    )

    logger.info(
        "Loading ZARR dataset..."
    )

    ds, temp, times = loader.load()

    logger.info(
        f"Temperature shape: {temp.shape}"
    )

    ########################################################
    # Split
    ########################################################

    (
        train,
        val,
        test,
        train_end,
        val_end
    ) = loader.split(
        temp
    )

    ########################################################
    # Normalize
    ########################################################

    (
        train,
        val,
        test,
        mean,
        std
    ) = loader.normalize(
        train,
        val,
        test
    )

    logger.info(
        f"Mean={mean:.6f}"
    )

    logger.info(
        f"Std={std:.6f}"
    )

    ########################################################
    # Time Features
    ########################################################

    (
        sin_hour,
        cos_hour,
        sin_doy,
        cos_doy
    ) = FeatureBuilder.build_time_features(
        times
    )

    lat_grid = (
        FeatureBuilder.build_latitude_feature(
            ds["latitude"].values,
            len(ds["longitude"])
        )
    )

    ########################################################
    # Train feature slices
    ########################################################

    train_sin_hour = (
        sin_hour[:train_end]
    )

    train_cos_hour = (
        cos_hour[:train_end]
    )

    train_sin_doy = (
        sin_doy[:train_end]
    )

    train_cos_doy = (
        cos_doy[:train_end]
    )

    ########################################################
    # Validation feature slices
    ########################################################

    val_sin_hour = (
        sin_hour[
            train_end:val_end
        ]
    )

    val_cos_hour = (
        cos_hour[
            train_end:val_end
        ]
    )

    val_sin_doy = (
        sin_doy[
            train_end:val_end
        ]
    )

    val_cos_doy = (
        cos_doy[
            train_end:val_end
        ]
    )

    ########################################################
    # Datasets
    ########################################################

    train_ds = ConvLSTMDataset(
        train,
        train_sin_hour,
        train_cos_hour,
        train_sin_doy,
        train_cos_doy,
        lat_grid,
        HISTORY,
        FORECAST
    )

    val_ds = ConvLSTMDataset(
        val,
        val_sin_hour,
        val_cos_hour,
        val_sin_doy,
        val_cos_doy,
        lat_grid,
        HISTORY,
        FORECAST
    )

    ########################################################
    # DataLoader Optimization
    ########################################################

    cpu_count = os.cpu_count()

    if cpu_count is None:
        cpu_count = 4

    num_workers = min(
        8,
        max(
            1,
            cpu_count // 2
        )
    )

    logger.info(
        f"CPU cores detected = {cpu_count}"
    )

    logger.info(
        f"DataLoader workers = {num_workers}"
    )

    ########################################################
    # Dataloaders
    ########################################################

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=True,
        prefetch_factor=4
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
        prefetch_factor=4
    )

    logger.info(
        f"Train samples = {len(train_ds)}"
    )

    logger.info(
        f"Validation samples = {len(val_ds)}"
    )

    logger.info(
        f"Train batches = {len(train_loader)}"
    )

    logger.info(
        f"Validation batches = {len(val_loader)}"
    )

    ########################################################
    # Model
    ########################################################

    model = ForecastConvLSTM(
        input_channels=6,
        hidden_dim=64,
        forecast_steps=FORECAST
    ).to(
        DEVICE
    )

    criterion = (
        torch.nn.MSELoss()
    )

    optimizer = (
        torch.optim.Adam(
            model.parameters(),
            lr=LR
        )
    )

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=DEVICE,
        logger=logger
    )

    ########################################################
    # Training
    ########################################################

    best_val_loss = 1.0e30

    training_start = time.time()

    logger.info(
        f"Training started "
        f"(epochs={EPOCHS}, "
        f"batch={BATCH_SIZE}, "
        f"device={DEVICE})"
    )

    os.makedirs(
        "checkpoints",
        exist_ok=True
    )

    for epoch in range(EPOCHS):

        epoch_start = time.time()

        train_loss = (
            trainer.train_epoch(
                train_loader
            )
        )

        val_loss = (
            trainer.validate(
                val_loader
            )
        )

        epoch_time = (
            time.time()
            - epoch_start
        )

        elapsed = (
            time.time()
            - training_start
        )

        avg_epoch = (
            elapsed
            / (epoch + 1)
        )

        eta = (
            avg_epoch
            * (
                EPOCHS
                - epoch
                - 1
            )
        )

        msg = (
            f"Epoch "
            f"{epoch+1:03d}/{EPOCHS} "
            f"Train={train_loss:.6f} "
            f"Val={val_loss:.6f} "
            f"EpochTime={epoch_time:.1f}s "
            f"ETA={eta/60:.1f} min"
        )

        print(
            msg,
            flush=True
        )

        logger.info(
            msg
        )

        ####################################################
        # Save best checkpoint
        ####################################################

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            timestamp = (
                datetime.now().strftime(
                    "%Y%m%d_%H%M%S"
                )
            )

            checkpoint_file = (
                f"checkpoints/"
                f"t2m_{MODEL_TYPE}_{timestamp}.pt"
            )

            torch.save(
                {
                    "model_type": MODEL_TYPE,
                    "model_state_dict": model.state_dict(),
                    "mean": float(mean),
                    "std": float(std),
                    "history": int(HISTORY),
                    "forecast": int(FORECAST),
                    "hidden_dim": model.hidden_dim,
                    "best_val_loss": float(val_loss)
                },
                checkpoint_file
            )

            logger.info(
                f"Checkpoint saved "
                f"({checkpoint_file}) "
                f"val={val_loss:.6f}"
            )

    ########################################################
    # End
    ########################################################

    total_minutes = (
        time.time()
        - training_start
    ) / 60.0

    logger.info(
        f"Training finished "
        f"runtime={total_minutes:.2f} min"
    )

    print(
        "Training complete."
    )


if __name__ == "__main__":
    main()

