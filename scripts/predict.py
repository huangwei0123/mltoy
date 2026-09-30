#!/usr/bin/env python

import argparse

import numpy as np
import pandas as pd
import torch
import xarray as xr

from models.forecast_unet import ForecastUNet
from utils.feature_builder import FeatureBuilder

from datetime import datetime

def log(msg):
    print(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}",
        flush=True
    )

class Predictor:

    def __init__(
        self,
        checkpoint_file,
        device="cpu"
    ):

        self.device = device

        print(
            f"Loading checkpoint: "
            f"{checkpoint_file}"
        )
        log(f"Loading checkpoint: {checkpoint_file}")

        checkpoint = torch.load(
            checkpoint_file,
            map_location=device,
            weights_only=False
        )

        self.mean = checkpoint["mean"]
        self.std = checkpoint["std"]

        self.history = checkpoint["history"]
        self.forecast = checkpoint["forecast"]

        self.model = ForecastUNet(
            input_channels=self.history + 5,
            forecast_steps=self.forecast
        ).to(device)

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.eval()

        print(
            f"Loaded model "
            f"(history={self.history}, "
            f"forecast={self.forecast})"
        )
        log(f"Loaded model history={self.history}, forecast={self.forecast}")

    def predict(self, x):

        with torch.no_grad():

            x = x.to(self.device)

            y = self.model(x)

        return y.cpu().numpy()


def build_input_tensor(
    ds,
    predictor,
    start_time=None
):

    temp = ds["t2m"].values.astype(
        np.float32
    )

    if "valid_time" in ds.coords:
        time_coord = "valid_time"
    elif "time" in ds.coords:
        time_coord = "time"
    else:
        raise ValueError(
            "No time coordinate found"
        )

    times = pd.to_datetime(
        ds[time_coord].values
    )

    history = predictor.history

    if start_time is None:

        current_idx = len(times) - 1

    else:

        init_time = pd.to_datetime(
            start_time,
            format="%Y%m%d-%H%M"
        )

        matches = np.where(
            times == init_time
        )[0]

        if len(matches) == 0:
            raise ValueError(
                f"Initialization time "
                f"{init_time} not found "
                f"in dataset"
            )

        current_idx = matches[0]

    if current_idx < history - 1:
        raise ValueError(
            "Not enough history available"
        )

    temp = (
        temp - predictor.mean
    ) / predictor.std

    temp_stack = temp[
        current_idx - history + 1:
        current_idx + 1
    ]

    (
        sin_hour,
        cos_hour,
        sin_doy,
        cos_doy
    ) = FeatureBuilder.build_time_features(
        times
    )

    nlat = temp.shape[1]
    nlon = temp.shape[2]

    sh = np.full(
        (1, nlat, nlon),
        sin_hour[current_idx],
        dtype=np.float32
    )

    ch = np.full(
        (1, nlat, nlon),
        cos_hour[current_idx],
        dtype=np.float32
    )

    sd = np.full(
        (1, nlat, nlon),
        sin_doy[current_idx],
        dtype=np.float32
    )

    cd = np.full(
        (1, nlat, nlon),
        cos_doy[current_idx],
        dtype=np.float32
    )

    lat_grid = (
        FeatureBuilder.build_latitude_feature(
            ds["latitude"].values,
            len(ds["longitude"])
        )
    )

    lat = lat_grid[np.newaxis, :, :]

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

    x = torch.tensor(
        x,
        dtype=torch.float32
    )

    x = x.unsqueeze(0)

    return x

def save_forecast(
    ds,
    forecast,
    output_file,
    predictor,
    start_time=None
):
    # Denormalize
    forecast = (
        forecast * predictor.std
        + predictor.mean
    )

    forecast = forecast[0]

    # Detect time coordinate
    if "valid_time" in ds.coords:
        time_coord = "valid_time"
    elif "time" in ds.coords:
        time_coord = "time"
    else:
        raise ValueError(
            "No time coordinate found "
            "(expected 'valid_time' or 'time')."
        )

    times_in = pd.to_datetime(
        ds[time_coord].values
    )

    if len(times_in) < 2:
        raise ValueError(
            "Need at least two timestamps "
            "to determine forecast interval."
        )

    # Compute all intervals
    # intervals = np.diff(times_in)

    # Use the most recent interval
    # freq = pd.Timedelta(intervals[-1])

    times_in = pd.to_datetime(
        ds[time_coord].values
    )

    if len(times_in) < 2:
        freq = pd.Timedelta(hours=6)
    else:
        intervals = np.diff(times_in)

        freq = pd.to_timedelta(
            intervals[-1]
        )

        if not np.all(
            intervals == intervals[0]
        ):
            print(
                "WARNING: irregular time spacing detected"
            )

    print(f"Detected interval: {freq}")

    # Warn about irregular spacing
    if not np.all(intervals == intervals[0]):
        print(
            "WARNING: Irregular time intervals detected. "
            "Using latest interval:",
            freq
        )
        log(
            f"Irregular intervals detected. "
            f"Using interval {freq}"
        )

    if start_time is None:
        last_time = pd.Timestamp(times_in[-1])
    else:
        last_time = pd.to_datetime(start_time, format="%Y%m%d-%H%M")

    print(f"Detected interval: {freq}")
    log(f"Detected interval: {freq}")

    forecast_times = pd.date_range(
        start=last_time + freq,
        periods=predictor.forecast,
        freq=freq
    )

    out_ds = xr.Dataset(
        data_vars={
            "t2m": (
                (
                    time_coord,
                    "latitude",
                    "longitude"
                ),
                forecast
            )
        },
        coords={
            time_coord: forecast_times,
            "latitude": ds["latitude"].values,
            "longitude": ds["longitude"].values
        }
    )

    # Useful metadata
    out_ds["t2m"].attrs.update({
        "long_name": "2 metre temperature",
        "units": "K"
    })

    out_ds.attrs.update({
        "model": "ForecastUNet",
        "history_steps": predictor.history,
        "forecast_steps": predictor.forecast
    })

    out_ds.to_netcdf(output_file)

    print(
        f"Forecast saved: {output_file}"
    )
    log(
        f"Forecast saved: {output_file}"
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        required=True
    )

    parser.add_argument(
        "--input",
        required=True
    )

    parser.add_argument(
        "--output",
        required=True
    )

    parser.add_argument(
        "--device",
        default="cpu"
    )

    parser.add_argument(
        "--start",
        default=None,
        help="Forecast initialization time YYYYMMDD-HHMM"
    )

    args = parser.parse_args()

    predictor = Predictor(
        args.checkpoint,
        device=args.device
    )

    print(
        f"Opening dataset: "
        f"{args.input}"
    )
    log(f"Opening dataset: {args.input}")

    ds = xr.open_zarr(
        args.input
    )

    x = build_input_tensor(
        ds,
        predictor,
        start_time=args.start
    )

    print(
        "Running forecast..."
    )
    log("Running forecast...")

    forecast = predictor.predict(x)

    print(
        f"Forecast shape: "
        f"{forecast.shape}"
    )
    log(f"Forecast shape: {forecast.shape}")

    save_forecast(
        ds,
        forecast,
        args.output,
        predictor,
        start_time=args.start
    )

    print("Done.")
    log("Done.")


if __name__ == "__main__":
    main()

