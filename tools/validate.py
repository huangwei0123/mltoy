# scripts/validate.py

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path


#########################################################################
# CONFIG
#########################################################################

FORECAST_FILE = (
    "forecasts/forecast_20260901_0000.nc"
)

TRUTH_DATASET = (
    "data/era5-t2m-5deg.zarr"
)

OUTPUT_CSV = (
    "forecasts/verification_metrics.csv"
)


#########################################################################
# LOAD FORECAST
#########################################################################

print("Loading forecast...")

fcst_ds = xr.open_dataset(
    FORECAST_FILE
)

forecast = fcst_ds["t2m_forecast"].values

valid_times = pd.to_datetime(
    fcst_ds["valid_time"].values
)

lead_times = (
    fcst_ds["lead_time"]
    .values
)

print("Forecast shape:")
print(forecast.shape)


#########################################################################
# LOAD ERA5 TRUTH
#########################################################################

print("\nLoading ERA5 dataset...")

truth_ds = xr.open_zarr(
    TRUTH_DATASET
)

print(truth_ds)

era5_times = pd.to_datetime(
    truth_ds["valid_time"].values
)


#########################################################################
# METRIC FUNCTIONS
#########################################################################

def rmse(fcst, obs):

    return np.sqrt(
        np.mean(
            (fcst - obs) ** 2
        )
    )


def mae(fcst, obs):

    return np.mean(
        np.abs(fcst - obs)
    )


def bias(fcst, obs):

    return np.mean(
        fcst - obs
    )


def acc(fcst, obs):

    fcst_anom = (
        fcst - np.mean(fcst)
    )

    obs_anom = (
        obs - np.mean(obs)
    )

    denom = np.sqrt(
        np.sum(fcst_anom**2)
        *
        np.sum(obs_anom**2)
    )

    if denom == 0:
        return np.nan

    return (
        np.sum(
            fcst_anom * obs_anom
        )
        / denom
    )


#########################################################################
# VERIFY LEAD TIMES
#########################################################################

results = []

print("\nVerification Results")
print("-" * 70)

for i, vt in enumerate(valid_times):

    matches = np.where(
        era5_times == vt
    )[0]

    if len(matches) == 0:

        print(
            f"Lead lead_times[{i:3d}]h "
            f"Missing truth: {vt}"
        )

        continue

    idx = matches[0]

    truth = (
        truth_ds["t2m"]
        .isel(valid_time=idx)
        .values
    )

    fcst = forecast[i]

    lead_rmse = rmse(
        fcst,
        truth
    )

    lead_mae = mae(
        fcst,
        truth
    )

    lead_bias = bias(
        fcst,
        truth
    )

    lead_acc = acc(
        fcst,
        truth
    )

    print(
        f"lead_times[{i:3d}]h "
        f"RMSE={lead_rmse:6.3f} "
        f"MAE={lead_mae:6.3f} "
        f"BIAS={lead_bias:7.3f} "
        f"ACC={lead_acc:6.3f}"
    )

    results.append(
        {
            "lead_hour":
                int(lead_times[i]),

            "valid_time":
                str(vt),

            "rmse":
                float(lead_rmse),

            "mae":
                float(lead_mae),

            "bias":
                float(lead_bias),

            "acc":
                float(lead_acc)
        }
    )


#########################################################################
# SAVE CSV
#########################################################################

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    OUTPUT_CSV,
    index=False
)

print("\nSaved:")
print(OUTPUT_CSV)

if len(results_df) > 0:

    print("\nSummary")

    print(
        results_df[
            ["rmse", "mae", "bias", "acc"]
        ].mean()
    )

else:

    print(
        "\nNo verification cases found."
    )
