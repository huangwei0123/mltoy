#!/usr/bin/env python

import sys

import numpy as np
import pandas as pd
import xarray as xr


class Evaluator:

    @staticmethod
    def rmse(pred, truth):
        return np.sqrt(
            np.mean(
                (pred - truth) ** 2
            )
        )

    @staticmethod
    def mae(pred, truth):
        return np.mean(
            np.abs(pred - truth)
        )

    @staticmethod
    def bias(pred, truth):
        return np.mean(
            pred - truth
        )

    @staticmethod
    def corr(pred, truth):

        p = pred.flatten()
        t = truth.flatten()

        mask = (
            np.isfinite(p)
            & np.isfinite(t)
        )

        return np.corrcoef(
            p[mask],
            t[mask]
        )[0, 1]

    @staticmethod
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

        return num / den

    @staticmethod
    def global_mean(field):

        return np.mean(
            field,
            axis=(1, 2)
        )


def main():

    forecast_file = sys.argv[1]
    truth_file = sys.argv[2]

    fcst = xr.open_dataset(
        forecast_file
    )

    truth = xr.open_dataset(
        truth_file
    )

    pred = fcst["t2m"].values
    obs = truth["t2m"].values

    if pred.shape != obs.shape:

        raise ValueError(
            f"Shape mismatch "
            f"{pred.shape} "
            f"vs "
            f"{obs.shape}"
        )

    print()
    print("=" * 70)
    print("OVERALL VERIFICATION")
    print("=" * 70)

    print(
        f"RMSE : "
        f"{Evaluator.rmse(pred, obs):.3f}"
    )

    print(
        f"MAE  : "
        f"{Evaluator.mae(pred, obs):.3f}"
    )

    print(
        f"BIAS : "
        f"{Evaluator.bias(pred, obs):.3f}"
    )

    print(
        f"CORR : "
        f"{Evaluator.corr(pred, obs):.3f}"
    )

    print(
        f"ACC  : "
        f"{Evaluator.acc(pred, obs):.3f}"
    )

    print()

    print("=" * 70)
    print("PER-LEAD SCORES")
    print("=" * 70)

    print(
        f"{'Lead':>6} "
        f"{'RMSE':>10} "
        f"{'MAE':>10} "
        f"{'BIAS':>10} "
        f"{'ACC':>10}"
    )

    nlead = pred.shape[0]

    for lead in range(nlead):

        p = pred[lead]
        t = obs[lead]

        clim = np.mean(
            obs,
            axis=0
        )

        p_anom = p - clim
        t_anom = t - clim

        acc_num = np.sum(
            p_anom * t_anom
        )

        acc_den = np.sqrt(
            np.sum(
                p_anom**2
            ) *
            np.sum(
                t_anom**2
            )
        )

        acc = (
            acc_num / acc_den
            if acc_den > 0
            else np.nan
        )

        print(
            f"{(lead+1)*6:6d} "
            f"{Evaluator.rmse(p,t):10.3f} "
            f"{Evaluator.mae(p,t):10.3f} "
            f"{Evaluator.bias(p,t):10.3f} "
            f"{acc:10.3f}"
        )

    print()

    print("=" * 70)
    print("GLOBAL MEAN TEMPERATURE")
    print("=" * 70)

    gm_pred = Evaluator.global_mean(
        pred
    )

    gm_obs = Evaluator.global_mean(
        obs
    )

    print()

    print(
        f"{'Lead':>6} "
        f"{'Forecast':>12} "
        f"{'Truth':>12} "
        f"{'Error':>12}"
    )

    for i in range(
        len(gm_pred)
    ):

        error = (
            gm_pred[i]
            - gm_obs[i]
        )

        print(
            f"{(i+1)*6:6d} "
            f"{gm_pred[i]:12.3f} "
            f"{gm_obs[i]:12.3f} "
            f"{error:12.3f}"
        )

    print()
    print("Done.")
    print()


if __name__ == "__main__":
    main()
