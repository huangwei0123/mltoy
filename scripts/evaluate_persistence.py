#!/usr/bin/env python

import argparse

import numpy as np
import pandas as pd
import xarray as xr


def rmse(fcst, truth):

    return np.sqrt(
        np.mean(
            (fcst - truth) ** 2
        )
    )


def mae(fcst, truth):

    return np.mean(
        np.abs(
            fcst - truth
        )
    )

def acc(fcst, truth):

    f = fcst.flatten()
    t = truth.flatten()

    f = f - f.mean()
    t = t - t.mean()

    return np.corrcoef( f, t)[0, 1]


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="ERA5 zarr"
    )

    parser.add_argument(
        "--start",
        required=True,
        help="YYYYMMDD-HHMM"
    )

    parser.add_argument(
        "--forecast",
        type=int,
        default=12
    )

    args = parser.parse_args()

    ds = xr.open_zarr(
        args.input
    )

    times = pd.to_datetime(
        ds["valid_time"].values
    )

    temp = (
        ds["t2m"]
        .values
        .astype(np.float32)
    )

    init_time = pd.to_datetime(
        args.start,
        format="%Y%m%d-%H%M"
    )

    idx = np.where(
        times == init_time
    )[0]

    if len(idx) == 0:

        raise ValueError(
            f"{init_time} not found"
        )

    idx = idx[0]

    ####################################################
    # Persistence forecast
    ####################################################

    current = temp[idx]

    print()
    print(
        f"{'Lead':>5} "
        f"{'RMSE':>12} "
        f"{'MAE':>12}"
        f"{'ACC':>12}"
    )

    print("-" * 35)

    rmses = []
    maes = []
    accs = []

    for lead in range( 1, args.forecast + 1):
        truth = temp[idx + lead]
        forecast = current

        r = rmse(forecast, truth)
        m = mae(forecast, truth)
        a = acc(forecast, truth)

        rmses.append(r)
        maes.append(m)
        accs.append(a)

        print(f"{lead:5d} {r:12.4f} {m:12.4f} {a:12.4f}")

    print()
    print(f"Mean RMSE = " f"{np.mean(rmses):.4f}")
    print(f"Mean MAE  = " f"{np.mean(maes):.4f}")
    print(f"Mean ACC  = " f"{np.mean(accs):.4f}")

if __name__ == "__main__":
    main()
