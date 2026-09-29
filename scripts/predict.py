# predict.py

import xarray as xr
import torch
import numpy as np

CHECKPOINT = "models/t2m_cpu.pt"
DATASET = "data/era5-t2m-5deg.zarr"

HISTORY = 8

ckpt = torch.load(
    CHECKPOINT,
    map_location="cpu"
)

ds = xr.open_zarr(DATASET)

temp = ds.t2m.values.astype(np.float32)

mean = ckpt["mean"]
std = ckpt["std"]

temp = (temp - mean) / std

latest = temp[-HISTORY:]

x = torch.tensor(
    latest
).unsqueeze(0)

class SimpleForecastCNN(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(8,64,3,padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64,64,3,padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64,12,1)
        )

    def forward(self,x):
        return self.net(x)

model = SimpleForecastCNN()

model.load_state_dict(
    ckpt["model"]
)

model.eval()

with torch.no_grad():

    forecast = model(x)

forecast = forecast.numpy()[0]

forecast = forecast * std + mean

print("Forecast shape:")

print(forecast.shape)
