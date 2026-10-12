#!/bin/bash

set -x

python -m scripts.train_cnn \
    --config configs/cnn.yaml
