For your problem, I'd preprocess the data into sequence-to-sequence training samples:

Input:
t-42h
t-36h
t-30h
t-24h
t-18h
t-12h
t-06h
t

Output:
t+06h
t+12h
t+18h
...
t+72h


assuming 6-hourly ERA5 data.

1. Load ERA5 NetCDF
import xarray as xr
import numpy as np

ds = xr.open_dataset("era5_t2m.nc")

print(ds)

t2m = ds["t2m"]   # (time, lat, lon)

# Convert K -> C if desired
t2m = t2m - 273.15

data = t2m.values.astype(np.float32)

print(data.shape)


Example:

(14600, 73, 144)

2. Handle Missing Values

ERA5 usually has none, but check anyway:

print(np.isnan(data).sum())


If needed:

mean_temp = np.nanmean(data)

data = np.nan_to_num(
    data,
    nan=mean_temp
)

3. Train/Validation/Test Split

Never shuffle weather data.

n = len(data)

train_end = int(n * 0.7)
val_end   = int(n * 0.85)

train = data[:train_end]
val   = data[train_end:val_end]
test  = data[val_end:]

4. Compute Training Statistics

Normalize using training data only.

mean = train.mean()
std  = train.std()

train = (train - mean) / std
val   = (val - mean) / std
test  = (test - mean) / std

print(mean, std)


Save these values:

np.save(
    "normalization.npy",
    {
        "mean": mean,
        "std": std
    }
)

5. Add Cyclical Time Features

Weather is highly seasonal.

times = ds.time.to_index()


Create:

dayofyear = times.dayofyear.values

sin_doy = np.sin(
    2*np.pi*dayofyear/365.25
)

cos_doy = np.cos(
    2*np.pi*dayofyear/365.25
)


For 6-hour data:

hour = times.hour.values

sin_hour = np.sin(
    2*np.pi*hour/24
)

cos_hour = np.cos(
    2*np.pi*hour/24
)

6. Build Samples

Configuration:

history = 8      # 48h
forecast = 12    # 72h


Data:

X = []
Y = []

for t in range(
        history,
        len(train)-forecast):

    x = train[
        t-history:t
    ]

    y = train[
        t:t+forecast
    ]

    X.append(x)
    Y.append(y)

X = np.array(X)
Y = np.array(Y)


Shapes:

X.shape

(N,8,73,144)

Y.shape

(N,12,73,144)

7. Add Channel Dimension

Most ConvLSTM implementations expect:

(batch,time,height,width,channels)


Convert:

X = X[..., np.newaxis]
Y = Y[..., np.newaxis]

print(X.shape)
print(Y.shape)


Output:

(N,8,73,144,1)

(N,12,73,144,1)

8. More Efficient Dataset Class (Recommended)

Avoid storing everything in RAM.

PyTorch
from torch.utils.data import Dataset
import torch


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

        x = x.unsqueeze(-1)
        y = y.unsqueeze(-1)

        return x, y


Usage:

train_ds = ERA5Dataset(
    train,
    history=8,
    forecast=12
)

9. Add Latitude as a Static Feature

This helps a lot.

Create:

lat = ds["latitude"].values


or

lat = ds["lat"].values


Normalize:

lat_norm = lat / 90.0


Expand:

lat_grid = np.repeat(
    lat_norm[:,None],
    len(ds.lon),
    axis=1
)


Shape:

73 x 144


Add as a second channel:

channel 0 = temperature
channel 1 = latitude


Many weather models do this.

10. Save as Zarr

For multi-year ERA5 datasets this is much faster than NetCDF.

ds.to_zarr(
    "era5_t2m.zarr",
    mode="w"
)


Then:

xr.open_zarr()


loads lazily and scales much better.

My V1 preprocessing pipeline

If I were building this myself, I'd start with:

Input history:
48 hours (8 timesteps)

Target:
72 hours (12 timesteps)

Variables:
- t2m
- latitude channel

Normalization:
global mean/std

Model:
ConvLSTM encoder-decoder

Loss:
MSE

Metric:
RMSE at
+6
+12
+24
+48
+72 hours


Then in V2 I'd add ERA5 u10, v10, surface_pressure, and dewpoint, which usually improves forecast skill much more than increasing model complexity.
