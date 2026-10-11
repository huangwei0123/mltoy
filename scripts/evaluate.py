#!/usr/bin/env python

import argparse
import os

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt

import pickle

from utils.metrics import rmse, mae, acc, get_climatology_field

def main():

    parser = argparse.ArgumentParser( description=( "Evaluate forecast against truth"))
    parser.add_argument("--forecast", required=True, help="Forecast NetCDF file")
    parser.add_argument("--truth", required=True, help="Truth NetCDF file")
    parser.add_argument("--outdir", default="forecasts")
    parser.add_argument("--climatology", required=True)
    args = parser.parse_args()

    with open(args.climatology, "rb") as f:
        climatology = pickle.load(f)

    print(f"Loading forecast: {args.forecast}")
    print(f"Loading truth: {args.truth}")

    fcst_ds = xr.open_dataset(args.forecast)

    truth_ds = xr.open_dataset(args.truth)

    ####################################################
    # Detect variable
    ####################################################

    if "t2m" not in fcst_ds:

        raise ValueError(
            f"t2m not found in "
            f"{args.forecast}"
        )

    if "t2m" not in truth_ds:

        raise ValueError(
            f"t2m not found in "
            f"{args.truth}"
        )

    fcst = (
        fcst_ds["t2m"]
        .values
        .astype(np.float32)
    )

    truth = (truth_ds["t2m"].values.astype(np.float32))

    ####################################################
    # Match lead times
    ####################################################

    nlead = min(fcst.shape[0], truth.shape[0])

    fcst = fcst[:nlead]
    truth = truth[:nlead]

    ####################################################
    # Metrics
    ####################################################

    results = []

    print()
    print( f"{'Lead':>5} {'RMSE':>12} {'MAE':>12} {'ACC':>12}")

    print("-" * 55)

    for lead in range(nlead):

        r = rmse(fcst[lead], truth[lead])
        m = mae(fcst[lead], truth[lead])

        valid_time = pd.to_datetime(truth_ds["valid_time"].values[ lead ])
        clim = get_climatology_field(climatology, valid_time)
        a = acc(fcst[lead], truth[lead], clim)

        print(f"{lead+1:5d} {r:12.4f} {m:12.4f} {a:12.4f}")

        results.append(
            {
                "lead": lead + 1,
                "rmse": r,
                "mae": m,
                "acc": a,
            }
        )

    ####################################################
    # Summary
    ####################################################

    rmse_mean = np.mean([r["rmse"] for r in results])
    mae_mean = np.mean([r["mae"] for r in results])

    print()
    print(f"Mean RMSE = {rmse_mean:.4f}")

    print(f"Mean MAE  = {mae_mean:.4f}")

    ####################################################
    # Save CSV
    ####################################################

    os.makedirs(args.outdir, exist_ok=True)

    model_name = os.path.splitext(
        os.path.basename(
            args.forecast
        )
    )[0]

    csv_file = os.path.join(
        args.outdir,
        f"{model_name}_metrics.csv"
    )

    pd.DataFrame(
        results
    ).to_csv(
        csv_file,
        index=False
    )

    ####################################################
    # Plot RMSE
    ####################################################

    plt.figure(figsize=(8, 4))

    plt.plot(
        [r["lead"] for r in results],
        [r["rmse"] for r in results],
        marker="o"
    )

    plt.grid(True)

    plt.xlabel("Forecast Lead")

    plt.ylabel("RMSE (K)")

    plt.title(f"RMSE vs Lead Time: {model_name}")

    png_file = os.path.join(args.outdir, f"{model_name}_rmse.png")

    plt.savefig(png_file, bbox_inches="tight")

    plt.close()

    print()
    print(f"Saved metrics: {csv_file}")

    print(f"Saved plot: {png_file}")


if __name__ == "__main__":
    main()
