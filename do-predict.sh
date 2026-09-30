#!/bin/bash

set -x

python -m scripts.predict \
    --checkpoint data/t2m_cpu.pt \
    --input data/era5-t2m-5deg.zarr \
    --start 20260901-0000 \
    --output forecasts/t2m_20260901_0000.nc

