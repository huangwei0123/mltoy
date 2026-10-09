import numpy as np
import pandas as pd
import xarray as xr


def save_forecast(
    ds,
    forecast,
    output_file,
    predictor,
    start_time=None
):

    forecast = (
        forecast * predictor.std
        + predictor.mean
    )

    forecast = forecast[0]

    # ConvLSTM:
    # (forecast, 1, lat, lon)
    # ->
    # (forecast, lat, lon)
    #
    if forecast.ndim == 4 and forecast.shape[1] == 1:
        forecast = forecast[:, 0]

    if "valid_time" in ds.coords:
        time_coord = "valid_time"
    elif "time" in ds.coords:
        time_coord = "time"
    else:
        raise ValueError(
            "No time coordinate found"
        )

    times_in = pd.to_datetime(
        ds[time_coord].values
    )

    if len(times_in) < 2:

        freq = pd.Timedelta(hours=6)

    else:

        intervals = np.diff(
            times_in
        )

        freq = pd.to_timedelta(
            intervals[-1]
        )

    if start_time is None:

        last_time = pd.Timestamp(
            times_in[-1]
        )

    else:

        last_time = pd.to_datetime(
            start_time,
            format="%Y%m%d-%H%M"
        )

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
            time_coord:
                forecast_times,

            "latitude":
                ds["latitude"].values,

            "longitude":
                ds["longitude"].values
        }
    )

    out_ds.attrs.update(
        {
            "model":
                predictor.model_type,

            "history_steps":
                predictor.history,

            "forecast_steps":
                predictor.forecast,
        }
    )

    out_ds.to_netcdf(
        output_file
    )

    print(
        f"Saved {output_file}"
    )
