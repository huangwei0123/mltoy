#!/bin/bash

set -x

#python -m scripts.train_unet

python -m scripts.train_unet_from_cnn \
    --train-zarr data/train.zarr \
    --valid-zarr data/valid.zarr \
    --cnn-checkpoint checkpoints/t2m_cnn.pt \
    --epochs 20
