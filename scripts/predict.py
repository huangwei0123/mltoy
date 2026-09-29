# scripts/predict.py

import numpy as np
import pandas as pd
import xarray as xr
import torch
import torch.nn as nn

from pathlib import Path


#########################################################################
# CONFIG
#########################################################################

CHECKPOINT = "models/t2m_cpu.pt"
DATASET = "data/era5-t2m-5deg.zarr"

OUTPUT_DIR = Path("forecasts")
OUTPUT_DIR.mkdir(exist_ok=True)


#########################################################################
# LOAD CHECKPOINT
#########################################################################

print("Loading checkpoint...")

ckpt = torch.load(
    CHECKPOINT,
    map_location="cpu",
    weights_only=False
)

mean = float(ckpt["mean"])
std = float(ckpt["std"])

HISTORY = int(ckpt["history"])
FORECAST = int(ckpt["forecast"])

print(f"HISTORY  = {HISTORY}")
print(f"FORECAST = {FORECAST}")
print(f"MEAN     = {mean}")
print(f"STD      = {std}")


#########################################################################
# MODEL
#########################################################################

INPUT_CHANNELS = HISTORY + 5


class ForecastCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(

            nn.Conv2d(
                INPUT_CHANNELS,
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
                128,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv2d(
                128,
                FORECAST,
                kernel_size=1
            )
        )

    def forward(self, x):

        return self.net(x)


#########################################################################
# LOAD DATA
#########################################################################

print("\nLoading dataset...")

ds = xr.open_zarr(DATASET)

print(ds)

temp = ds["t2m"].values.astype(np.float32)

times = ds["valid_time"].to_index()

latitudes = ds["latitude"].values
longitudes = ds["longitude"].values

init_time = pd.Timestamp(
    ds["valid_time"].values[-1]
)

print("\nForecast init time:")
print(init_time)

print("\nTemperature shape:")
print(temp.shape)


#########################################################################
# NORMALIZE
#########################################################################

temp_norm = (
    temp - mean
) / std


#########################################################################
# HISTORY TEMPERATURES
#########################################################################

INIT_TIME = "2026-09-01 00:00:00"

init_time = pd.Timestamp(INIT_TIME)

idx = times.get_loc(init_time)

temp_stack = temp_norm[
    idx - HISTORY + 1:
    idx + 1
]

current_time = times[idx]

#########################################################################
# TIME FEATURES
#########################################################################

hour = current_time.hour
doy = current_time.dayofyear

sin_hour = np.sin(
    2.0 * np.pi * hour / 24.0
).astype(np.float32)

cos_hour = np.cos(
    2.0 * np.pi * hour / 24.0
).astype(np.float32)

sin_doy = np.sin(
    2.0 * np.pi * doy / 365.25
).astype(np.float32)

cos_doy = np.cos(
    2.0 * np.pi * doy / 365.25
).astype(np.float32)


#########################################################################
# LATITUDE FEATURE
#########################################################################

lat_norm = (
    latitudes / 90.0
).astype(np.float32)

lat_grid = np.repeat(
    lat_norm[:, None],
    len(longitudes),
    axis=1
)


#########################################################################
# CONSTANT FEATURE MAPS
#########################################################################

nlat = len(latitudes)
nlon = len(longitudes)

sh = np.full(
    (1, nlat, nlon),
    sin_hour,
    dtype=np.float32
)

ch = np.full(
    (1, nlat, nlon),
    cos_hour,
    dtype=np.float32
)

sd = np.full(
    (1, nlat, nlon),
    sin_doy,
    dtype=np.float32
)

cd = np.full(
    (1, nlat, nlon),
    cos_doy,
    dtype=np.float32
)

lat = lat_grid[np.newaxis, :, :]


#########################################################################
# BUILD INPUT
#########################################################################

x_input = np.concatenate(
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

print("\nInput channels:")
print(x_input.shape)

# Expected:
# (13, 36, 72)

x = torch.tensor(
    x_input,
    dtype=torch.float32
).unsqueeze(0)

print("\nModel input shape:")
print(x.shape)

# Expected:
# (1, 13, 36, 72)


#########################################################################
# LOAD MODEL
#########################################################################

print("\nLoading model...")

model = ForecastCNN()

model.load_state_dict(
    ckpt["model_state_dict"]
)

model.eval()

print("Model loaded successfully")


#########################################################################
# FORECAST
#########################################################################

print("\nRunning forecast...")

with torch.no_grad():

    forecast = model(x)

forecast = forecast.numpy()[0]

forecast = (
    forecast * std
) + mean

print("\nForecast shape:")
print(forecast.shape)

# Expected:
# (12, 36, 72)


#########################################################################
# LEAD TIMES
#########################################################################

lead_hours = np.arange(
    6,
    FORECAST * 6 + 1,
    6
)

valid_times = (
    init_time
    + pd.to_timedelta(
        lead_hours,
        unit="h"
    )
)


#########################################################################
# CREATE DATASET
#########################################################################

forecast_ds = xr.Dataset(
    {
        "t2m_forecast": (
            (
                "lead_time",
                "latitude",
                "longitude"
            ),
            forecast
        )
    },
    coords={
        "lead_time": lead_hours,
        "valid_time": (
            "lead_time",
            valid_times
        ),
        "latitude": latitudes,
        "longitude": longitudes,
    }
)

forecast_ds["t2m_forecast"].attrs = {
    "units": "K",
    "long_name": "2 metre temperature forecast"
}

forecast_ds.attrs["forecast_init_time"] = str(
    init_time
)

forecast_ds.attrs["history_steps"] = HISTORY
forecast_ds.attrs["forecast_steps"] = FORECAST


#########################################################################
# SAVE NETCDF
#########################################################################

timestamp = init_time.strftime(
    "%Y%m%d_%H%M"
)

outfile = (
    OUTPUT_DIR
    / f"forecast_{timestamp}.nc"
)

forecast_ds.to_netcdf(outfile)

print("\nForecast saved:")
print(outfile)

print("\nForecast Dataset:")
print(forecast_ds)
