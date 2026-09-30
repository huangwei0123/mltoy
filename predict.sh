#!/bin/bash

set -x

if [[ ! -f data/truth.nc ]]; then
    python -m scripts.make_truth \
        --input data/era5-t2m-5deg.zarr \
        --start 20260901-0000 \
        --forecast_steps 12 \
        --output data/truth.nc
fi

python -m scripts.predict \
    --checkpoint data/t2m_cpu.pt \
    --input data/era5-t2m-5deg.zarr \
    --start 20260901-0000 \
    --output forecasts/t2m_20260901_0000.nc

python -m scripts.evaluate \
    forecasts/t2m_20260901_0000.nc \
    data/truth.nc

python tools/plot_forecast.py
python tools/plot_metrics.py forecasts/t2m_20260901_0000.nc data/truth.nc

