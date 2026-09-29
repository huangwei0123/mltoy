import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path


##############################################################################
# CONFIG
##############################################################################

FORECAST_FILE = "forecasts/forecast_20260901_0000.nc"

TRUTH_DATASET = "data/era5-t2m-5deg.zarr"

OUTPUT_DIR = Path("forecasts/plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


##############################################################################
# LOAD DATA
##############################################################################

print("Loading forecast...")

fcst_ds = xr.open_dataset(
    FORECAST_FILE
)

forecast = fcst_ds["t2m_forecast"].values

lead_times = fcst_ds["lead_time"].values

valid_times = pd.to_datetime(
    fcst_ds["valid_time"].values
)

lat = fcst_ds["latitude"].values
lon = fcst_ds["longitude"].values

print("Loading truth...")

truth_ds = xr.open_zarr(
    TRUTH_DATASET
)

era5_times = pd.to_datetime(
    truth_ds["valid_time"].values
)


##############################################################################
# PLOT LOOP
##############################################################################

for i, valid_time in enumerate(valid_times):

    matches = np.where(
        era5_times == valid_time
    )[0]

    if len(matches) == 0:

        print(
            f"No truth found for {valid_time}"
        )

        continue

    idx = matches[0]

    truth = (
        truth_ds["t2m"]
        .isel(valid_time=idx)
        .values
    )

    fcst = forecast[i]

    error = fcst - truth

    rmse = np.sqrt(
        np.mean(error ** 2)
    )

    ##########################################################################
    # FIGURE
    ##########################################################################

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 5)
    )

    #
    # Truth
    #

    im0 = axes[0].pcolormesh(
        lon,
        lat,
        truth,
        shading="auto",
        cmap="coolwarm"
    )

    axes[0].set_title(
        f"ERA5 Truth\n{valid_time}"
    )

    plt.colorbar(
        im0,
        ax=axes[0],
        label="K"
    )

    #
    # Forecast
    #

    im1 = axes[1].pcolormesh(
        lon,
        lat,
        fcst,
        shading="auto",
        cmap="coolwarm",
        vmin=np.min(truth),
        vmax=np.max(truth)
    )

    axes[1].set_title(
        f"Forecast\nLead lead_times[{i}] h"
    )

    plt.colorbar(
        im1,
        ax=axes[1],
        label="K"
    )

    #
    # Error
    #

    vmax = np.max(
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
        f"Forecast Error\nRMSE={rmse:.2f} K"
    )

    plt.colorbar(
        im2,
        ax=axes[2],
        label="K"
    )

    plt.tight_layout()

    outfile = (
        OUTPUT_DIR
        / f"lead_lead_times_{i:03d}.png"
    )

    plt.savefig(
        outfile,
        dpi=150,
        bbox_inches="tight"
    )

    # plt.show()
    plt.close()

    print(
        f"Saved {outfile}"
    )

print("Done.")
