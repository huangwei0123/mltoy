#!/usr/bin/env python

import argparse
import pickle

import xarray as xr
import pandas as pd

from pandas import Timedelta

from forecasts.predictor import Predictor
from forecasts.input_builder import build_input_tensor
from forecasts.netcdf_writer import save_forecast


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        required=True
    )

    parser.add_argument(
        "--climatology",
        required=True
    )

    parser.add_argument(
        "--input",
        required=True
    )

    parser.add_argument(
        "--output",
        required=True
    )

    parser.add_argument(
        "--device",
        default="cpu"
    )

    parser.add_argument(
        "--start",
        default=None
    )

    args = parser.parse_args()

    predictor = Predictor(
        args.checkpoint,
        device=args.device
    )

    print(
        "predictor mean =",
        predictor.mean
    )

    print(
        "predictor std =",
        predictor.std
    )

    ds = xr.open_zarr(
        args.input
    )

    with open(
        args.climatology,
        "rb"
    ) as f:

        climatology = pickle.load(f)

    x = build_input_tensor(
        ds,
        predictor,
        args.start,
        climatology
    )

    print(
        "input shape:",
        x.shape
    )

    ####################################################
    # Predict normalized anomaly
    ####################################################

    anomaly_forecast = predictor.predict(x)[0]

    print(
        "normalized anomaly min/max:",
        anomaly_forecast.min(),
        anomaly_forecast.max()
    )

    ####################################################
    # Denormalize anomaly
    ####################################################

    anomaly_forecast = (
        anomaly_forecast
        * predictor.std
        + predictor.mean
    )

    print(
        "physical anomaly min/max:",
        anomaly_forecast.min(),
        anomaly_forecast.max()
    )

    ####################################################
    # Add climatology
    ####################################################

    start_time = pd.to_datetime(
        args.start,
        format="%Y%m%d-%H%M"
    )

    temperature_forecast = anomaly_forecast.copy()

    for lead in range(
        predictor.forecast
    ):

        valid_time = (
            start_time
            + Timedelta(
                hours=6 * (lead + 1)
            )
        )

        doy = valid_time.dayofyear

        cycle = (
            valid_time.hour // 6
        )

        temperature_forecast[lead] += (
            climatology[
                (doy, cycle)
            ]
        )

    print(
        "temperature_forecast shape =",
        temperature_forecast.shape
    )

    print(
        "temperature_forecast min/max:",
        temperature_forecast.min(),
        temperature_forecast.max()
    )

    ####################################################
    # Save physical temperature
    ####################################################

    save_forecast(
        ds,
        temperature_forecast,
        args.output,
        predictor,
        args.start
    )


if __name__ == "__main__":
    main()
