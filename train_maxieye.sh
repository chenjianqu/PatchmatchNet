#!/usr/bin/env bash

MVS_TRAINING="/media/cjq/新加卷/datasets/OpenSourceDatasets/training/dtu_PatchmatchNet"

# Train on converted DTU training set
python train_maxieye.py --batch_size 3 --epochs 8  --image_max_dim 480 "$@"

