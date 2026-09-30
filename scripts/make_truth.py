#!/usr/bin/env python

import argparse

import pandas as pd
import xarray as xr


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="Input ERA5 Zarr dataset"
    )

    parser.add_argument(
        "--start",
        required=True,
        help="Forecast initialization time YYYYMMDD-HHMM"
    )

    parser.add_argument(
        "--forecast_steps",
        type=int,
        default=12
    )

    parser.add_argument(
        "--output",
        required=True
    )

    args = parser.parse_args()

    ds = xr.open_zarr(
        args.input
    )

    if "valid_time" in ds.coords:
        time_coord = "valid_time"
    elif "time" in ds.coords:
        time_coord = "time"
    else:
        raise ValueError(
            "Dataset contains neither "
            "'valid_time' nor 'time'"
        )

    init_time = pd.Timestamp(
        pd.to_datetime(
            args.start,
            format="%Y%m%d-%H%M",
            utc=True
        )
    )

    times = pd.to_datetime(
        ds[time_coord].values,
        utc=True
    )

    if init_time not in times:
        raise ValueError(
            f"Initialization time "
            f"{init_time} not found "
            f"in dataset"
        )

    idx = times.get_loc(init_time)

    if idx + args.forecast_steps >= len(times):
        raise ValueError(
            "Not enough future truth data "
            "available in dataset"
        )

    truth = ds.isel(
        {
            time_coord: slice(
                idx + 1,
                idx + 1 + args.forecast_steps
            )
        }
    )

    truth.to_netcdf(
        args.output
    )

    print()
    print(
        f"Initialization time : {init_time}"
    )

    print(
        f"Truth steps         : "
        f"{args.forecast_steps}"
    )

    print(
        f"Truth start         : "
        f"{truth[time_coord].values[0]}"
    )

    print(
        f"Truth end           : "
        f"{truth[time_coord].values[-1]}"
    )

    print(
        f"Saved: {args.output}"
    )
    print()


if __name__ == "__main__":
    main()
