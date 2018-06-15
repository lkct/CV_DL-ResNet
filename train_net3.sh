#!/bin/bash

set -e

python -u train_net3.py --depth 110 --batch-size 256 --num-examples 50000 --gpus 0 --workspace 1024 \
    --lr 0.1 --wd 0.0001 --num-classes 10 # --model-load-epoch 0 --retrain
