#!/bin/bash

set -e

python -u train_densenet.py --depth 76 --batch-size 256 --num-examples 50000 --gpus 0 --workspace 1024 \
    --lr 0.1 --wd 0.0001 --num-classes 10 # --model-load-epoch 0 --retrain
# depth 112 oom, (no memonger)