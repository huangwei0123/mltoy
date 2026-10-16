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
from utils.config import Config
from datasets.forecast_dataset import ForecastDataset

from models.forecast_cnn import ForecastCNN

from training.trainer import Trainer

from datetime import datetime

import argparse

STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
MODEL_TYPE = "cnn"
LOG_FILE = f"logs/train_{MODEL_TYPE}_{STAMP}.log"

def main(yamlfile="configs/cnn.yaml"):
    logger = LoggerFactory.create(LOG_FILE)
    logger.info("Starting CNN training run")
    logger.info(f"yamlfile: {yamlfile}")
    cfg = Config(yamlfile)

    ########################################################
    # Load ERA5
    ########################################################

    loader = ERA5DataLoader(cfg.dataset["path"])

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
            len(ds["longitude"])
        )
    )

    (sin_lon_grid, cos_lon_grid) = (
        FeatureBuilder.build_longitude_features(
            ds["longitude"].values,
            len(ds["latitude"])
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
        sin_lon_grid,
        cos_lon_grid,
        cfg.model["history"],
        cfg.model["forecast"]
    )

    val_ds = ForecastDataset(
        val,
        val_sin_hour,
        val_cos_hour,
        val_sin_doy,
        val_cos_doy,
        lat_grid,
        sin_lon_grid,
        cos_lon_grid,
        cfg.model["history"],
        cfg.model["forecast"]
    )

    ########################################################
    # Dataloaders
    ########################################################

    train_loader = DataLoader(train_ds, batch_size=cfg.training["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.training["batch_size"], shuffle=False)

    ########################################################
    # Model
    ########################################################

    model = ForecastCNN(
        input_channels=cfg.model["history"] + 7,
        forecast_steps=cfg.model["forecast"]
    ).to(
        cfg.training["device"]
    )

    criterion = (torch.nn.MSELoss())
    optimizer = (torch.optim.Adam(model.parameters(), lr=cfg.training["learning_rate"]))

    start_epoch = 0
    best_val_loss = 1.0e30

########################################################
# Resume from checkpoint
########################################################

    resume_checkpoint = cfg.checkpoint.get("resume", None)
    logger.info(f"resume_checkpoint: {resume_checkpoint}")
    if resume_checkpoint is not None:
        logger.info(f"Loading checkpoint: " f"{resume_checkpoint}")
        ckpt = torch.load(resume_checkpoint, map_location=cfg.training["device"], weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        best_val_loss = ckpt.get("best_val_loss", 1.0e30)
        logger.info(f"Resuming from epoch={start_epoch}, best_val={best_val_loss:.6f}")

        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            logger.info("Optimizer state restored")

        if "schedular_state_dict" in ckpt:
            schedularr.load_state_dict(ckpt["schedular_state_dict"])
            logger.info("Schedular state restored")

########################################################
# Training
########################################################

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=cfg.training["device"],
        logger=logger
    )

    training_start = time.time()

    logger.info(f"Training started (epoch={cfg.training["epochs"]}, batch={cfg.training["batch_size"]}, device={cfg.training["device"]})")

    new_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_file = f"checkpoints/t2m_{MODEL_TYPE}_{new_stamp}.pt"

    os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)

    for epoch in range(start_epoch, cfg.training["epochs"]):
        epoch_start = time.time()
        train_loss = (trainer.train_epoch(train_loader))
        val_loss = (trainer.validate(val_loader))
        epoch_time = (time.time() - epoch_start )
        elapsed = (time.time() - training_start )
        avg_epoch = (elapsed / (epoch + 1) )
        eta = (avg_epoch * ( cfg.training["epochs"] - epoch - 1 ) )
        msg = (
            f"Epoch "
            f"{epoch+1:03d}/{cfg.training["epochs"]} "
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
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=2,
                min_lr=1e-6,
            )

        torch.save(
            {
                "model_type": f"{MODEL_TYPE}",
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "mean": float(mean),
                "std": float(std),
                "history": int(cfg.model["history"]),
                "forecast": int(cfg.model["forecast"]),
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cnn.yaml")
    args = parser.parse_args()

    main(yamlfile=args.config)
