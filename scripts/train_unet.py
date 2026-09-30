import numpy as np
import xarray as xr
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import logging
import sys
import time
from datetime import datetime

#########################################################################
# LOGGING
#########################################################################

# def log(msg):
#     print(
#         f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}",
#         flush=True
#     )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[
        logging.FileHandler(
            "logs/train_t2m_cpu.log"
        ),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def log(msg):
    logger.info(msg)

#########################################################################
# CONFIG
#########################################################################

DATASET = "data/era5-t2m-5deg.zarr"

HISTORY = 8
FORECAST = 12

BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-3

DEVICE = "cpu"

#########################################################################
# LOAD DATA
#########################################################################

print("Loading ZARR dataset...")
log("Loading ZARR dataset...")

ds = xr.open_zarr(DATASET)

temp = ds["t2m"].values.astype(np.float32)

times = ds["valid_time"].to_index()

print("Temperature shape:", temp.shape)
sinfo = f"Temperature shape: {temp.shape}"
log(sinfo)

#########################################################################
# TIME FEATURES
#########################################################################

hours = times.hour.values

sin_hour = np.sin(
    2 * np.pi * hours / 24.0
).astype(np.float32)

cos_hour = np.cos(
    2 * np.pi * hours / 24.0
).astype(np.float32)

doy = times.dayofyear.values

sin_doy = np.sin(
    2 * np.pi * doy / 365.25
).astype(np.float32)

cos_doy = np.cos(
    2 * np.pi * doy / 365.25
).astype(np.float32)

#########################################################################
# LATITUDE FEATURE
#########################################################################

latitudes = ds["latitude"].values

lat_norm = (
    latitudes / 90.0
).astype(np.float32)

lat_grid = np.repeat(
    lat_norm[:, None],
    len(ds["longitude"]),
    axis=1
)

#########################################################################
# TRAIN/VAL/TEST SPLIT
#########################################################################

n = len(temp)

train_end = int(0.70 * n)
val_end = int(0.85 * n)

train = temp[:train_end]
val = temp[train_end:val_end]
test = temp[val_end:]

#########################################################################
# NORMALIZATION
#########################################################################

mean = train.mean()
std = train.std()

train = (train - mean) / std
val = (val - mean) / std
test = (test - mean) / std

print("Mean =", mean)
print("Std  =", std)

sinfo = f"Mean = {mean}"
log(sinfo)
sinfo = f"Std  = {std}"
log(sinfo)

#########################################################################
# SPLIT TIME FEATURES
#########################################################################

train_sin_hour = sin_hour[:train_end]
val_sin_hour = sin_hour[train_end:val_end]

train_cos_hour = cos_hour[:train_end]
val_cos_hour = cos_hour[train_end:val_end]

train_sin_doy = sin_doy[:train_end]
val_sin_doy = sin_doy[train_end:val_end]

train_cos_doy = cos_doy[:train_end]
val_cos_doy = cos_doy[train_end:val_end]

#########################################################################
# DATASET
#########################################################################

class ForecastDataset(Dataset):

    def __init__(
        self,
        temp,
        sin_hour,
        cos_hour,
        sin_doy,
        cos_doy,
        lat_grid,
        history,
        forecast
    ):

        self.temp = temp

        self.sin_hour = sin_hour
        self.cos_hour = cos_hour

        self.sin_doy = sin_doy
        self.cos_doy = cos_doy

        self.lat_grid = lat_grid

        self.history = history
        self.forecast = forecast

        self.nlat = temp.shape[1]
        self.nlon = temp.shape[2]

    def __len__(self):

        return (
            len(self.temp)
            - self.history
            - self.forecast
        )

    def __getitem__(self, idx):

        #################################################################
        # Temperature history
        #################################################################

        temp_stack = self.temp[
            idx:
            idx + self.history
        ]

        #################################################################
        # Current timestamp
        #################################################################

        current = idx + self.history - 1

        #################################################################
        # Time channels
        #################################################################

        sh = np.full(
            (1, self.nlat, self.nlon),
            self.sin_hour[current],
            dtype=np.float32
        )

        ch = np.full(
            (1, self.nlat, self.nlon),
            self.cos_hour[current],
            dtype=np.float32
        )

        sd = np.full(
            (1, self.nlat, self.nlon),
            self.sin_doy[current],
            dtype=np.float32
        )

        cd = np.full(
            (1, self.nlat, self.nlon),
            self.cos_doy[current],
            dtype=np.float32
        )

        #################################################################
        # Latitude channel
        #################################################################

        lat = self.lat_grid[np.newaxis, :, :]

        #################################################################
        # INPUT
        #################################################################

        x = np.concatenate(
            [
                temp_stack,
                sh,
                ch,
                sd,
                cd,
                lat
            ],
            axis=0
        )

        #################################################################
        # TARGET
        #################################################################

        y = self.temp[
            current + 1:
            current + 1 + self.forecast
        ]

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )


#########################################################################
# DATALOADERS
#########################################################################

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

#########################################################################
# MODEL
#########################################################################

class DoubleConv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_ch,
                out_ch,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

INPUT_CHANNELS = HISTORY + 5


class ForecastUNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.enc1 = DoubleConv(
            INPUT_CHANNELS,
            64
        )

        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(
            64,
            128
        )

        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConv(
            128,
            256
        )

        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(
            256,
            512
        )

        self.up3 = nn.ConvTranspose2d(
            512,
            256,
            kernel_size=2,
            stride=2
        )

        self.dec3 = DoubleConv(
            512,
            256
        )

        self.up2 = nn.ConvTranspose2d(
            256,
            128,
            kernel_size=2,
            stride=2
        )

        self.dec2 = DoubleConv(
            256,
            128
        )

        self.up1 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=2,
            stride=2
        )

        self.dec1 = DoubleConv(
            128,
            64
        )

        self.out_conv = nn.Conv2d(
            64,
            FORECAST,
            kernel_size=1
        )

    def match_size(
        self,
        x,
        ref
    ):

        if x.shape[-2:] != ref.shape[-2:]:

            x = F.interpolate(
                x,
                size=ref.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        return x

    def forward(self, x):

        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool1(e1)
        )

        e3 = self.enc3(
            self.pool2(e2)
        )

        b = self.bottleneck(
            self.pool3(e3)
        )

        d3 = self.up3(b)

        d3 = self.match_size(
            d3,
            e3
        )

        d3 = torch.cat(
            [d3, e3],
            dim=1
        )

        d3 = self.dec3(d3)

        d2 = self.up2(d3)

        d2 = self.match_size(
            d2,
            e2
        )

        d2 = torch.cat(
            [d2, e2],
            dim=1
        )

        d2 = self.dec2(d2)

        d1 = self.up1(d2)

        d1 = self.match_size(
            d1,
            e1
        )

        d1 = torch.cat(
            [d1, e1],
            dim=1
        )

        d1 = self.dec1(d1)

        return self.out_conv(d1)

model = ForecastUNet().to(DEVICE)

#########################################################################
# LOSS + OPTIMIZER
#########################################################################

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR
)

#########################################################################
# TRAIN LOOP
#########################################################################

best_val_loss = 1e9

training_start = time.time()

log(
    f"Training starting "
    f"(epochs={EPOCHS}, "
    f"batch_size={BATCH_SIZE}, "
    f"device={DEVICE})"
)

for epoch in range(EPOCHS):

    epoch_start = time.time()

    log(
        f"Starting epoch "
        f"{epoch + 1}/{EPOCHS}"
    )

    model.train()

    train_loss = 0.0

    num_batches = len(train_loader)

    for batch_idx, (x, y) in enumerate(train_loader):

        if batch_idx % 10 == 0:

            elapsed = time.time() - epoch_start

            log(
                f"Epoch {epoch+1}/{EPOCHS} "
                f"Batch {batch_idx}/{num_batches} "
                f"Elapsed={elapsed:.2f}s"
            )

        x = x.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()

        pred = model(x)

        loss = criterion(pred, y)

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    #####################################################################

    model.eval()

    val_loss = 0.0

    with torch.no_grad():

        for x, y in val_loader:

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            pred = model(x)

            loss = criterion(pred, y)

            val_loss += loss.item()

    val_loss /= len(val_loader)

    print(
        f"Epoch {epoch+1:03d} "
        f"Train={train_loss:.6f} "
        f"Val={val_loss:.6f}"
    )

    sinfo = f"Epoch {epoch+1:03d}, Train={train_loss:.6f}, Val={val_loss:.6f}"
    log(sinfo)

    epoch_time = time.time() - epoch_start

    total_elapsed = time.time() - training_start

    avg_epoch_time = total_elapsed / (epoch + 1)

    remaining_epochs = EPOCHS - (epoch + 1)

    eta_seconds = avg_epoch_time * remaining_epochs

    log(
        f"Epoch {epoch+1:03d} "
        f"Train={train_loss:.6f} "
        f"Val={val_loss:.6f} "
        f"EpochTime={epoch_time:.2f}s "
        f"ETA={eta_seconds/60:.2f} min"
    )


    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "mean": mean,
                "std": std,

                "history": HISTORY,
                "forecast": FORECAST
            },
            "data/t2m_cpu.pt"
        )

        print("Checkpoint saved.")
        log(
            f"Checkpoint saved "
            f"(val_loss={val_loss:.6f})"
        )

#########################################################################
# DONE
#########################################################################

print("Training finished.")
total_time = time.time() - training_start

log(
    f"Training finished. "
    f"Total runtime="
    f"{total_time/60:.2f} minutes"
)
