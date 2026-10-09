#!/usr/bin/env python

import os
import time
import pickle

import torch

from torch.utils.data import DataLoader

from utils.config import *
from utils.logger import LoggerFactory

from utils.data_loader import ERA5DataLoader
from utils.feature_builder import FeatureBuilder
from utils.climatology import ClimatologyBuilder
from datasets.forecast_dataset import ForecastDataset

from models.forecast_cnn import ForecastCNN

from training.trainer import Trainer

from datetime import datetime

STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
MODEL_TYPE = "cnn"
LOG_FILE = f"logs/train_{MODEL_TYPE}_{STAMP}.log"

def main():

    logger = LoggerFactory.create(LOG_FILE)

    logger.info("Starting CNN training run")

    ########################################################
    # Load ERA5
    ########################################################

    loader = ERA5DataLoader(DATASET)

    logger.info("Loading ZARR dataset...")

    ds, temp, times = loader.load()

    logger.info("Building climatology...")

    climatology = (ClimatologyBuilder.compute( temp, times))

    logger.info("Creating anomalies...")

    temp = (ClimatologyBuilder.anomaly( temp, times, climatology))

    logger.info(f"Temperature shape: {temp.shape}")

    os.makedirs("checkpoints", exist_ok=True)

    with open("checkpoints/climatology.pkl", "wb") as f:
        pickle.dump(climatology, f)

    ########################################################
    # Split
    ########################################################

    (train, val, test, train_end, val_end) = loader.split(temp)

    ########################################################
    # Normalize
    ########################################################

    (train, val, test, mean, std) = loader.normalize( train, val, test)

    logger.info(f"Mean={mean:.6f}")

    logger.info(f"Std={std:.6f}")

    ########################################################
    # Features
    ########################################################

    (sin_hour, cos_hour, sin_doy, cos_doy) = FeatureBuilder.build_time_features(times)

    lat_grid = (
        FeatureBuilder.build_latitude_feature(
            ds["latitude"].values,
            len(
                ds["longitude"]
            )
        )
    )

    ########################################################
    # Train feature slices
    ########################################################

    train_sin_hour = (sin_hour[:train_end])
    train_cos_hour = (cos_hour[:train_end])
    train_sin_doy = (sin_doy[:train_end])
    train_cos_doy = (cos_doy[:train_end])

    ########################################################
    # Validation slices
    ########################################################

    val_sin_hour = (sin_hour[ train_end:val_end ])
    val_cos_hour = (cos_hour[ train_end:val_end ])
    val_sin_doy = (sin_doy[ train_end:val_end ])
    val_cos_doy = (cos_doy[ train_end:val_end ])

    ########################################################
    # Datasets
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
    # Dataloaders
    ########################################################

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    ########################################################
    # Model
    ########################################################

    model = ForecastCNN(
        input_channels=HISTORY + 5,
        forecast_steps=FORECAST
    ).to(
        DEVICE
    )

    criterion = (torch.nn.MSELoss())
    optimizer = (torch.optim.Adam(model.parameters(), lr=LR))

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

    new_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_file = f"checkpoints/t2m_{MODEL_TYPE}_{new_stamp}.pt"

    os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)

    for epoch in range(EPOCHS):
        epoch_start = time.time()
        train_loss = (trainer.train_epoch(train_loader))
        val_loss = (trainer.validate(val_loader))
        epoch_time = (time.time() - epoch_start )
        elapsed = (time.time() - training_start )
        avg_epoch = (elapsed / (epoch + 1) )
        eta = (avg_epoch * ( EPOCHS - epoch - 1 ) )
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

        new_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_file = f"checkpoints/t2m_{MODEL_TYPE}_epoch{epoch+1}_{new_stamp}.pt"
        else:
            checkpoint_file = f"checkpoints/skip_t2m_{MODEL_TYPE}_epoch{epoch+1}_{new_stamp}.pt"

        torch.save(
            {
                "model_type": f"{MODEL_TYPE}",
                "model_state_dict": model.state_dict(),
                "mean": float(mean),
                "std": float(std),
                "history": int(HISTORY),
                "forecast": int(FORECAST),
                "epoch": int(epoch+1),
                "best_val_loss": float(val_loss)
            },
            checkpoint_file
        )

        logger.info(f"Checkpoint saved ({checkpoint_file}), val={val_loss:.6f}")

    total_minutes = (time.time() - training_start) / 60.0

    logger.info(f"Training finished runtime={total_minutes:.2f} min")

    print("Training complete.")


if __name__ == "__main__":
    main()
