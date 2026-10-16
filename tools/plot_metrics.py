#!/usr/bin/env python

import sys

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

import pickle
from utils.metrics import rmse, mae, acc, bias, get_climatology_field

def plot_skill_vs_persistence(
    leads,
    rmse_model,
    acc_model,
    rmse_persist,
    acc_persist,
    outdir
):
    rmse_skill = 1.0 - np.asarray(rmse_model) / np.asarray(rmse_persist)
    acc_gain = np.asarray(acc_model) - np.asarray(acc_persist)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    ax[0].plot(leads, rmse_skill, marker="o", lw=2)
    ax[0].axhline(0.0, color="k", ls="--", lw=1)
    ax[0].set_title("RMSE Skill vs Persistence")
    ax[0].set_xlabel("Lead")
    ax[0].set_ylabel("Skill Score")
    ax[0].grid(True, alpha=0.3)

    ax[1].plot(leads, acc_gain, marker="o", lw=2)
    ax[1].axhline(0.0, color="k", ls="--", lw=1)
    ax[1].set_title("ACC Improvement vs Persistence")
    ax[1].set_xlabel("Lead")
    ax[1].set_ylabel("ACC Difference")
    ax[1].grid(True, alpha=0.3)

    fig.tight_layout()

    outfile = outdir / "skill_vs_persistence.png"
    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    print(f"Saved {outfile}")

##############################################################################
# MAIN
##############################################################################

def main():

    if len(sys.argv) != 3:
        print("Usage:\n" "python tools/plot_metrics.py " "forecast.nc truth.nc")
        sys.exit(1)

    # metrics = np.load(args.persistence)
    metrics = np.load("data/persistence_metrics.npz")
    persistence_rmse = metrics["rmse"]
    persistence_acc = metrics["acc"]

    with open( "checkpoints/climatology.pkl", "rb") as f:
        climatology = pickle.load(f)

    forecast_file = sys.argv[1]
    truth_file = sys.argv[2]

    print("Loading forecast...")

    fcst_ds = xr.open_dataset(forecast_file)

    print("Loading truth...")

    truth_ds = xr.open_dataset(truth_file)
    pred = fcst_ds["t2m"].values
    obs = truth_ds["t2m"].values

    if pred.shape != obs.shape:
        raise ValueError(
            f"Shape mismatch: "
            f"{pred.shape} "
            f"vs "
            f"{obs.shape}"
        )

    output_dir = Path("output/plots")

    output_dir.mkdir(parents=True, exist_ok=True)

    leads = []
    rmse_list = []
    mae_list = []
    bias_list = []
    acc_list = []

    nlead = pred.shape[0]

    print()
    print("Computing metrics...")
    print()
    print( f"{'Lead':>5} {'RMSE':>12} {'MAE':>12} {'BIAS':>12} {'ACC':>12}")
    print("-" * 55)

    for lead in range(nlead):
        lead_hr = (lead + 1) * 6

        p = pred[lead]
        t = obs[lead]

        leads.append(lead_hr)
  
        valid_time = pd.to_datetime( truth_ds["valid_time"].values[ lead ])
        clim = get_climatology_field( climatology, valid_time)

        r = rmse(p, t)
        m = mae(p, t)
        b = bias(p, t)
        a = acc( p, t, clim)

        print(f"{lead+1:5d} {r:12.4f} {m:12.4f} {b:12.4} {a:12.4f}")

        rmse_list.append(r)
        mae_list.append(m)
        bias_list.append(b)
        acc_list.append(a)

    plot_skill_vs_persistence(
        leads,
        rmse_list,
        acc_list,
        persistence_rmse,
        persistence_acc,
        output_dir
    )

    ##########################################################################
    # RMSE / MAE
    ##########################################################################

    plt.figure(figsize=(8, 5))

    plt.plot(leads, rmse_list, marker="o", linewidth=2, label="RMSE")

    plt.plot(leads, mae_list, marker="s", linewidth=2, label="MAE")

    plt.xlabel("Forecast Lead (hours)")
    plt.ylabel("Temperature Error (°C)")

    plt.title("Forecast Error vs Lead Time")

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    outfile = (output_dir / "rmse_mae_vs_lead.png")

    plt.savefig(outfile, dpi=150)

    plt.show()
    plt.close()

    print(f"Saved {outfile}")

    ##########################################################################
    # ACC
    ##########################################################################

    plt.figure( figsize=(8, 5))

    plt.plot(leads, acc_list, marker="o", linewidth=2, color="darkgreen")

    plt.xlabel("Forecast Lead (hours)")

    plt.ylabel("ACC")

    plt.ylim(0.0, 1.0)

    plt.title("Anomaly Correlation Coefficient")
    plt.grid(True)

    plt.tight_layout()

    outfile = (output_dir / "acc_vs_lead.png")

    plt.savefig(outfile, dpi=150)

    plt.show()
    plt.close()

    print(f"Saved {outfile}")

    ##########################################################################
    # Bias
    ##########################################################################

    plt.figure(figsize=(8, 5))

    plt.plot(leads, bias_list, marker="o", linewidth=2, color="red")

    plt.axhline(0, color="black", linestyle="--")

    plt.xlabel("Forecast Lead (hours)")
    plt.ylabel("Bias (°C)")
    plt.title("Forecast Bias vs Lead Time")

    plt.grid(True)

    plt.tight_layout()

    outfile = (output_dir / "bias_vs_lead.png")

    plt.savefig(outfile, dpi=150)

    plt.show()
    plt.close()

    print(f"Saved {outfile}")

    print()
    
if __name__ == "__main__":
    main()
