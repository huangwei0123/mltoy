#!/bin/bash

set -x

if [[ ! -f data/truth.nc ]]; then
    python -m scripts.make_truth \
        --input data/era5-t2m-5deg.zarr \
        --start 20260901-0000 \
        --forecast_steps 12 \
        --output data/truth.nc
fi

epoch=18
#model_type=unet
model_type=cnn
#model_type=fno
#model_type=graphcast
#model_type=convlstm
ckpt_stamp=20261010_161112
fcst_stamp=20260901-0000
ckpt_file=checkpoints/t2m_${model_type}_epoch${epoch}_${ckpt_stamp}.pt
output=output/t2m_${fcst_stamp}_${model_type}_${ckpt_stamp}.nc

if [[ ! -f ${output} ]]; then
    python -m scripts.predict \
        --checkpoint ${ckpt_file} \
        --climatology checkpoints/climatology.pkl \
        --input data/era5-t2m-5deg.zarr \
        --start ${fcst_stamp} \
        --output ${output}
fi

python -m scripts.evaluate \
  --forecast ${output} \
  --truth data/truth.nc \
  --climatology checkpoints/climatology.pkl \
  --outdir output/plot_${model_type}_${ckpt_stamp}

python tools/plot_forecast.py \
  --forecast ${output} \
  --truth data/truth.nc \
  --outdir output/plot_${model_type}_${ckpt_stamp}

python tools/plot_metrics.py ${output} data/truth.nc

python -m scripts.evaluate_persistence \
  --input data/era5-t2m-5deg.zarr \
  --climatology checkpoints/climatology.pkl \
  --start 20260901-0000 \
  --forecast 12
