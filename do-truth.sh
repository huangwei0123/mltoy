#!/bin/bash

set -x

python -m scripts.make_truth \
--input data/era5-t2m-5deg.zarr \
--start 20260901-0000 \
--forecast_steps 12 \
--output data/truth.nc

