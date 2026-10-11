import numpy as np
import pandas as pd
import torch
import pickle

from utils.feature_builder import FeatureBuilder


def build_input_tensor(
    ds,
    predictor,
    start_time=None,
    climatology=None
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
                f"{init_time} not found"
            )

        current_idx = matches[0]

        if climatology is not None:

            anomaly = np.empty_like(temp)

            for i, t in enumerate(times):

                doy = t.dayofyear
                cycle = t.hour // 6

                anomaly[i] = (
                    temp[i]
                    -
                    climatology[(doy, cycle)]
                )

            temp = anomaly

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

    lat = lat_grid[np.newaxis]

    (sin_lon_grid, cos_lon_grid) = (
        FeatureBuilder.build_longitude_features(
            ds["longitude"].values,
            len(ds["latitude"])
        )
    )

    sin_lon = sin_lon_grid[np.newaxis]
    cos_lon = cos_lon_grid[np.newaxis]

    if predictor.model_type == "convlstm":

        feat_hist = np.empty(
            (history, 5, nlat, nlon),
            dtype=np.float32
        )

        feat_hist[:, 0] = sin_hour[
            current_idx - history + 1:
            current_idx + 1
        ][:, None, None]

        feat_hist[:, 1] = cos_hour[
            current_idx - history + 1:
            current_idx + 1
        ][:, None, None]

        feat_hist[:, 2] = sin_doy[
            current_idx - history + 1:
            current_idx + 1
        ][:, None, None]

        feat_hist[:, 3] = cos_doy[
            current_idx - history + 1:
            current_idx + 1
        ][:, None, None]

        feat_hist[:, 4] = lat_grid[None]

        temp_hist = temp_stack[:, None]

        x = np.concatenate(
            [
                temp_hist,
                feat_hist
            ],
            axis=1
        )

        print("input shape =", x.shape)

        return torch.tensor(
            x,
            dtype=torch.float32
        ).unsqueeze(0)

    else:

        x = np.concatenate(
            [
                temp_stack,
                sh,
                ch,
                sd,
                cd,
                lat,
                sin_lon,
                cos_lon
            ],
            axis=0
        )

        return torch.tensor(
            x,
            dtype=torch.float32
        ).unsqueeze(0)

