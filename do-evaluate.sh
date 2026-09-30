#!/bin/bash

set -x

python -m scripts.evaluate \
   forecasts/t2m_fcst.nc \
   data/truth.nc
