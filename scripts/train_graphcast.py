#!/usr/bin/env python

import os
import time

import torch

from torch.utils.data import DataLoader

from utils.config import *
from utils.logger import LoggerFactory

from utils.data_loader import ERA5DataLoader
from utils.feature_builder import FeatureBuilder

from datasets.forecast_dataset import ForecastDataset

from models.forecast_graphcast import (
    ForecastGraphCast
)

from training.trainer import Trainer

from datetime import datetime


STAMP = datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)

MODEL_TYPE = "graphcast"

LOG_FILE = (
    f"logs/train_{MODEL_TYPE}_{STAMP}.log"
)


def main():

    logger = LoggerFactory.create(
        LOG_FILE
    )

    logger.info(
        "Starting GraphCast training run"
    )

    ########################################################
    # Load data
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
    # Time features
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
    # Train slices
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
    # Validation slices
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
    # Dataset
    ########################################################

    train_ds = ForecastDataset(
        train,
        train_sin_hour,
        train_cos_hour,
        train_sin_doy,
        train_cos_doy,
        lat_grid,
        HISTORY,
        FORECAST
    )

    val_ds = ForecastDataset(
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
    # Very memory-friendly dataloaders
    ########################################################

    train_loader = DataLoader(
        train_ds,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    ########################################################
    # Lightweight GraphCast
    ########################################################

    model = ForecastGraphCast(
        input_channels=HISTORY + 5,
        forecast_steps=FORECAST,
        hidden_dim=32,
        num_layers=4
    ).to(
        DEVICE
    )

    ########################################################
    # Loss / optimizer
    ########################################################

    criterion = (
        torch.nn.MSELoss()
    )

    optimizer = (
        torch.optim.AdamW(
            model.parameters(),
            lr=LR,
            weight_decay=1e-4
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
        f"device={DEVICE})"
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
            elapsed / (epoch + 1)
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

        print(msg)
        logger.info(msg)

        ####################################################
        # Save best checkpoint
        ####################################################

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            checkpoint_file = (
                f"checkpoints/"
                f"t2m_{MODEL_TYPE}_"
                f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
            )

            os.makedirs(
                os.path.dirname(
                    checkpoint_file
                ),
                exist_ok=True
            )

            torch.save(
                {
                    "model_type": MODEL_TYPE,
                    "model_state_dict":
                        model.state_dict(),

                    "hidden_dim": 32,
                    "num_layers": 4,

                    "mean": float(mean),
                    "std": float(std),

                    "history":
                        int(HISTORY),

                    "forecast":
                        int(FORECAST),

                    "best_val_loss":
                        float(val_loss)
                },
                checkpoint_file
            )

            logger.info(
                f"Checkpoint saved "
                f"({checkpoint_file}) "
                f" val={val_loss:.6f}"
            )

    total_minutes = (
        time.time()
        - training_start
    ) / 60.0

    logger.info(
        f"Training finished "
        f"runtime={total_minutes:.2f} minutes"
    )

    print(
        "Training complete."
    )


if __name__ == "__main__":
    main()
