import torch

from torch.utils.data import DataLoader

from utils.config import *
from utils.logger import LoggerFactory
from utils.data_loader import ERA5DataLoader
from utils.feature_builder import FeatureBuilder

from datasets.forecast_dataset import ForecastDataset

from models.forecast_unet import ForecastUNet

from training.trainer import Trainer


logger = LoggerFactory.create(
    LOG_FILE
)

loader = ERA5DataLoader(
    DATASET
)

ds, temp, times = loader.load()

(
    train,
    val,
    test,
    train_end,
    val_end
) = loader.split(temp)

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

train_ds = ForecastDataset(
    train,
    sin_hour[:train_end],
    cos_hour[:train_end],
    sin_doy[:train_end],
    cos_doy[:train_end],
    lat_grid,
    HISTORY,
    FORECAST
)

val_ds = ForecastDataset(
    val,
    sin_hour[train_end:val_end],
    cos_hour[train_end:val_end],
    sin_doy[train_end:val_end],
    cos_doy[train_end:val_end],
    lat_grid,
    HISTORY,
    FORECAST
)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False
)

model = ForecastUNet(
    HISTORY + 5,
    FORECAST
).to(DEVICE)

criterion = torch.nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR
)

trainer = Trainer(
    model,
    criterion,
    optimizer,
    DEVICE,
    logger
)

for epoch in range(EPOCHS):

    train_loss = trainer.train_epoch(
        train_loader
    )

    val_loss = trainer.validate(
        val_loader
    )

    print(
        f"Epoch {epoch+1:03d} "
        f"Train={train_loss:.6f} "
        f"Val={val_loss:.6f}"
    )

