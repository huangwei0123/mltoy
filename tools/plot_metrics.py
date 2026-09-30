#!/usr/bin/env python

import sys

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from pathlib import Path


##############################################################################
# METRICS
##############################################################################

def rmse(pred, truth):

    return np.sqrt(
        np.mean(
            (pred - truth) ** 2
        )
    )


def mae(pred, truth):

    return np.mean(
        np.abs(pred - truth)
    )


def bias(pred, truth):

    return np.mean(
        pred - truth
    )


def acc(pred, truth):

    clim = np.mean(
        truth,
        axis=0,
        keepdims=True
    )

    p_anom = pred - clim
    t_anom = truth - clim

    num = np.sum(
        p_anom * t_anom
    )

    den = np.sqrt(
        np.sum(
            p_anom ** 2
        )
        *
        np.sum(
            t_anom ** 2
        )
    )

    if den == 0:
        return np.nan

    return num / den


##############################################################################
# MAIN
##############################################################################

def main():

    if len(sys.argv) != 3:

        print(
            "Usage:\n"
            "python tools/plot_metrics.py "
            "forecast.nc truth.nc"
        )

        sys.exit(1)

    forecast_file = sys.argv[1]
    truth_file = sys.argv[2]

    print("Loading forecast...")

    fcst_ds = xr.open_dataset(
        forecast_file
    )

    print("Loading truth...")

    truth_ds = xr.open_dataset(
        truth_file
    )

    pred = fcst_ds["t2m"].values
    obs = truth_ds["t2m"].values

    if pred.shape != obs.shape:

        raise ValueError(
            f"Shape mismatch: "
            f"{pred.shape} "
            f"vs "
            f"{obs.shape}"
        )

    output_dir = Path(
        "forecasts/plots"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    leads = []
    rmse_list = []
    mae_list = []
    bias_list = []
    acc_list = []

    nlead = pred.shape[0]

    print()
    print("Computing metrics...")
    print()

    for lead in range(nlead):

        lead_hr = (lead + 1) * 6

        p = pred[lead]
        t = obs[lead]

        leads.append(
            lead_hr
        )

        rmse_list.append(
            rmse(p, t)
        )

        mae_list.append(
            mae(p, t)
        )

        bias_list.append(
            bias(p, t)
        )

        clim = np.mean(
            obs,
            axis=0
        )

        p_anom = p - clim
        t_anom = t - clim

        num = np.sum(
            p_anom * t_anom
        )

        den = np.sqrt(
            np.sum(
                p_anom**2
            )
            *
            np.sum(
                t_anom**2
            )
        )

        acc_list.append(
            num / den
            if den > 0
            else np.nan
        )

    ##########################################################################
    # RMSE / MAE
    ##########################################################################

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        leads,
        rmse_list,
        marker="o",
        linewidth=2,
        label="RMSE"
    )

    plt.plot(
        leads,
        mae_list,
        marker="s",
        linewidth=2,
        label="MAE"
    )

    plt.xlabel(
        "Forecast Lead (hours)"
    )

    plt.ylabel(
        "Temperature Error (°C)"
    )

    plt.title(
        "Forecast Error vs Lead Time"
    )

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    outfile = (
        output_dir /
        "rmse_mae_vs_lead.png"
    )

    plt.savefig(
        outfile,
        dpi=150
    )

    plt.show()
    plt.close()

    print(
        f"Saved {outfile}"
    )

    ##########################################################################
    # ACC
    ##########################################################################

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        leads,
        acc_list,
        marker="o",
        linewidth=2,
        color="darkgreen"
    )

    plt.xlabel(
        "Forecast Lead (hours)"
    )

    plt.ylabel(
        "ACC"
    )

    plt.ylim(
        0.0,
        1.0
    )

    plt.title(
        "Anomaly Correlation Coefficient"
    )

    plt.grid(True)

    plt.tight_layout()

    outfile = (
        output_dir /
        "acc_vs_lead.png"
    )

    plt.savefig(
        outfile,
        dpi=150
    )

    plt.show()
    plt.close()

    print(
        f"Saved {outfile}"
    )

    ##########################################################################
    # Bias
    ##########################################################################

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        leads,
        bias_list,
        marker="o",
        linewidth=2,
        color="red"
    )

    plt.axhline(
        0,
        color="black",
        linestyle="--"
    )

    plt.xlabel(
        "Forecast Lead (hours)"
    )

    plt.ylabel(
        "Bias (°C)"
    )

    plt.title(
        "Forecast Bias vs Lead Time"
    )

    plt.grid(True)

    plt.tight_layout()

    outfile = (
        output_dir /
        "bias_vs_lead.png"
    )

    plt.savefig(
        outfile,
        dpi=150
    )

    plt.show()
    plt.close()

    print(
        f"Saved {outfile}"
    )

    print()
    
if __name__ == "__main__":
    main()
