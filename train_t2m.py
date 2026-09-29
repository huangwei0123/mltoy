import xarray as xr
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

##################################################
# CONFIG
##################################################

NC_FILE = "data/era5-t2m.nc"

HISTORY = 8
FORECAST = 12

BATCH_SIZE = 4
EPOCHS = 30

##################################################
# LOAD DATA
##################################################

print("Loading ERA5 ...")

ds = xr.open_dataset(NC_FILE)

# Kelvin -> Celsius
t2m = ds["t2m"] - 273.15

# ------------------------------------------------
# VERY IMPORTANT
# Use lower resolution first
# ------------------------------------------------

t2m = t2m[:, ::4, ::4]

data = t2m.values.astype(np.float32)

print("Shape:", data.shape)

##################################################
# SPLIT
##################################################

n = len(data)

train_end = int(n * 0.7)
val_end = int(n * 0.85)

train_data = data[:train_end]
val_data = data[train_end:val_end]
test_data = data[val_end:]

##################################################
# NORMALIZATION
##################################################

mean = train_data.mean()
std = train_data.std()

train_data = (train_data - mean) / std
val_data = (val_data - mean) / std
test_data = (test_data - mean) / std

print("mean =", mean)
print("std  =", std)

##################################################
# DATASET
##################################################

class ERA5Dataset(Dataset):

    def __init__(
        self,
        data,
        history=8,
        forecast=12
    ):
        self.data = data
        self.history = history
        self.forecast = forecast

    def __len__(self):
        return (
            len(self.data)
            - self.history
            - self.forecast
        )

    def __getitem__(self, idx):

        x = self.data[
            idx:
            idx+self.history
        ]

        y = self.data[
            idx+self.history:
            idx+self.history+self.forecast
        ]

        x = torch.tensor(
            x,
            dtype=torch.float32
        )

        y = torch.tensor(
            y,
            dtype=torch.float32
        )

        return x, y

##################################################
# DATALOADER
##################################################

train_ds = ERA5Dataset(
    train_data,
    HISTORY,
    FORECAST
)

val_ds = ERA5Dataset(
    val_data,
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

##################################################
# SIMPLE CNN FORECASTER
##################################################

class TempForecaster(nn.Module):

    def __init__(
        self,
        history,
        forecast
    ):
        super().__init__()

        self.history = history
        self.forecast = forecast

        self.encoder = nn.Sequential(

            nn.Conv2d(
                history,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv2d(
                128,
                256,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU()
        )

        self.decoder = nn.Sequential(

            nn.Conv2d(
                256,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv2d(
                128,
                forecast,
                kernel_size=1
            )
        )

    def forward(self, x):

        z = self.encoder(x)

        y = self.decoder(z)

        return y

##################################################
# MODEL
##################################################

model = TempForecaster(
    HISTORY,
    FORECAST
).to(DEVICE)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4
)

##################################################
# TRAIN
##################################################

for epoch in range(EPOCHS):

    model.train()

    train_loss = 0

    for x, y in train_loader:

        x = x.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()

        pred = model(x)

        loss = criterion(
            pred,
            y
        )

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    model.eval()

    val_loss = 0

    with torch.no_grad():

        for x, y in val_loader:

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            pred = model(x)

            loss = criterion(pred, y)

            val_loss += loss.item()

    print(
        f"Epoch {epoch+1:03d} "
        f"Train={train_loss/len(train_loader):.6f} "
        f"Val={val_loss/len(val_loader):.6f}"
    )

##################################################
# SAVE
##################################################

torch.save(
    {
        "model": model.state_dict(),
        "mean": mean,
        "std": std
    },
    "t2m_forecast.pt"
)

print("Saved t2m_forecast.pt")

