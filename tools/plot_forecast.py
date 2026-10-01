#!/usr/bin/env python

import argparse
import os

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from pathlib import Path

##############################################################################
# CONFIG
##############################################################################

parser = argparse.ArgumentParser( description=( "Evaluate forecast against truth"))

parser.add_argument("--forecast", required=True, help="Forecast NetCDF file")

parser.add_argument("--truth", required=True, help="Truth NetCDF file")

parser.add_argument("--outdir", default="forecasts/plots")

args = parser.parse_args()

print(f"Loading forecast: {args.forecast}")

print(f"Loading truth: {args.truth}")

FORECAST_FILE = args.forecast

TRUTH_FILE = args.truth

OUTPUT_DIR = Path(args.outdir)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

##############################################################################
# LOAD DATA
##############################################################################

print("Loading forecast...")

fcst_ds = xr.open_dataset(FORECAST_FILE)

print("Loading truth...")

truth_ds = xr.open_dataset(
    TRUTH_FILE
)

forecast = fcst_ds["t2m"].values
truth_all = truth_ds["t2m"].values

valid_times = pd.to_datetime(
    fcst_ds["valid_time"].values
)

lat = fcst_ds["latitude"].values
lon = fcst_ds["longitude"].values

nlead = forecast.shape[0]

print(
    f"Forecast shape: {forecast.shape}"
)

print(
    f"Truth shape: {truth_all.shape}"
)

##############################################################################
# GLOBAL COLOR LIMITS
##############################################################################

temp_min = min(
    np.nanmin(forecast),
    np.nanmin(truth_all)
)

temp_max = max(
    np.nanmax(forecast),
    np.nanmax(truth_all)
)

##############################################################################
# PLOT LOOP
##############################################################################

for i in range(nlead):

    truth = truth_all[i]

    fcst = forecast[i]

    error = fcst - truth

    rmse = np.sqrt(
        np.nanmean(
            error ** 2
        )
    )

    mae = np.nanmean(
        np.abs(error)
    )

    bias = np.nanmean(
        error
    )

    lead_hr = (i + 1) * 6

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 5)
    )

    ##########################################################################
    # TRUTH
    ##########################################################################

    im0 = axes[0].pcolormesh(
        lon,
        lat,
        truth,
        shading="auto",
        cmap="coolwarm",
        vmin=temp_min,
        vmax=temp_max
    )

    axes[0].set_title(
        f"ERA5 Truth\n"
        f"{valid_times[i]}"
    )

    plt.colorbar(
        im0,
        ax=axes[0],
        label="°C"
    )

    ##########################################################################
    # FORECAST
    ##########################################################################

    im1 = axes[1].pcolormesh(
        lon,
        lat,
        fcst,
        shading="auto",
        cmap="coolwarm",
        vmin=temp_min,
        vmax=temp_max
    )

    axes[1].set_title(
        f"Forecast\n"
        f"Lead +{lead_hr} h"
    )

    plt.colorbar(
        im1,
        ax=axes[1],
        label="°C"
    )

    ##########################################################################
    # ERROR
    ##########################################################################

    vmax = np.nanmax(
        np.abs(error)
    )

    im2 = axes[2].pcolormesh(
        lon,
        lat,
        error,
        shading="auto",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax
    )

    axes[2].set_title(
        f"Error\n"
        f"RMSE={rmse:.2f}°C  "
        f"MAE={mae:.2f}°C  "
        f"BIAS={bias:.2f}°C"
    )

    plt.colorbar(
        im2,
        ax=axes[2],
        label="°C"
    )

    ##########################################################################

    plt.tight_layout()

    outfile = (f"{args.outdir}/lead_{lead_hr:03d}h.png")

    plt.savefig(outfile, dpi=300, bbox_inches="tight")

    plt.close()

    print(f"Saved {outfile}")

print("Done.")
