#!/usr/bin/env python

import argparse

import xarray as xr

from forecasts.predictor import Predictor
from forecasts.input_builder import build_input_tensor
from forecasts.netcdf_writer import save_forecast

# from forecasts import (
#     Predictor,
#     build_input_tensor,
#     save_forecast,
# )

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
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

    ds = xr.open_zarr(
        args.input
    )

    x = build_input_tensor(
        ds,
        predictor,
        args.start
    )

    forecast = predictor.predict(
        x
    )

    save_forecast(
        ds,
        forecast,
        args.output,
        predictor,
        args.start
    )


if __name__ == "__main__":
    main()
