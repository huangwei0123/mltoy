# train_t2m_cpu.py

import xarray as xr
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

#########################################################
# CONFIG
#########################################################

DATASET = "data/era5-t2m-5deg.zarr"

HISTORY = 8      # 48 h
FORECAST = 12    # 72 h

EPOCHS = 20
BATCH_SIZE = 16
LR = 1e-3

DEVICE = "cpu"

#########################################################
# LOAD
#########################################################

print("Loading zarr...")

ds = xr.open_zarr(DATASET)

temp = ds["t2m"].values.astype(np.float32)

print(temp.shape)

#########################################################
# SPLIT
#########################################################

n = len(temp)

train_end = int(0.7 * n)
val_end = int(0.85 * n)

train = temp[:train_end]
val = temp[train_end:val_end]
test = temp[val_end:]

#########################################################
# NORMALIZE
#########################################################

mean = train.mean()
std = train.std()

train = (train - mean) / std
val = (val - mean) / std
test = (test - mean) / std

print("mean =", mean)
print("std  =", std)

#########################################################
# DATASET
#########################################################

class ForecastDataset(Dataset):

    def __init__(
        self,
        data,
        history,
        forecast
    ):
        self.data = data
        self.history = history
        self.forecast = forecast

    def __len__(self):
        return len(self.data) - self.history - self.forecast

    def __getitem__(self, idx):

        x = self.data[
            idx:
            idx + self.history
        ]

        y = self.data[
            idx + self.history:
            idx + self.history + self.forecast
        ]

        return (
            torch.tensor(x),
            torch.tensor(y)
        )

#########################################################
# DATALOADERS
#########################################################

train_loader = DataLoader(
    ForecastDataset(
        train,
        HISTORY,
        FORECAST
    ),
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    ForecastDataset(
        val,
        HISTORY,
        FORECAST
    ),
    batch_size=BATCH_SIZE
)

#########################################################
# MODEL
#########################################################

class SimpleForecastCNN(nn.Module):

    def __init__(
        self,
        history,
        forecast
    ):
        super().__init__()

        self.net = nn.Sequential(

            nn.Conv2d(
                history,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv2d(
                64,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv2d(
                64,
                forecast,
                kernel_size=1
            )
        )

    def forward(self, x):

        return self.net(x)

model = SimpleForecastCNN(
    HISTORY,
    FORECAST
)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR
)

#########################################################
# TRAIN
#########################################################

best_val = 999999

for epoch in range(EPOCHS):

    model.train()

    train_loss = 0

    for x, y in train_loader:

        optimizer.zero_grad()

        pred = model(x)

        loss = criterion(
            pred,
            y
        )

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    #####################################################
    # VALIDATION
    #####################################################

    model.eval()

    val_loss = 0

    with torch.no_grad():

        for x, y in val_loader:

            pred = model(x)

            loss = criterion(
                pred,
                y
            )

            val_loss += loss.item()

    val_loss /= len(val_loader)

    print(
        f"Epoch {epoch+1:03d} "
        f"train={train_loss:.5f} "
        f"val={val_loss:.5f}"
    )

    if val_loss < best_val:

        best_val = val_loss

        torch.save(
            {
                "model": model.state_dict(),
                "mean": mean,
                "std": std,
            },
            "models/t2m_cpu.pt"
        )

        print("saved")

print("training complete")
