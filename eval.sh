#!/usr/bin/env bash

CHECKPOINT_FILE="./checkpoints/params_000007.ckpt"

# test on DTU's evaluation set
#DTU_TESTING="/home/dtu/"
#python eval.py --scan_list ./lists/dtu/test.txt --input_folder=$DTU_TESTING --output_folder=$DTU_TESTING \
#--checkpoint_path $CHECKPOINT_FILE --num_views 5 --image_max_dim 1600 --geo_mask_thres 3 --photo_thres 0.8 "$@"

# -------------------------------------------------------------------------------------
# test on eth3d benchmark
ETH3D_TESTING="/media/cjq/新加卷/datasets/OpenSourceDatasets/test_data/eth3d/"
python eval.py --scan_list ./lists/eth3d/test.txt --input_folder=$ETH3D_TESTING --output_folder=$ETH3D_TESTING \
--checkpoint_path $CHECKPOINT_FILE --num_views 7 --image_max_dim 2688 --geo_mask_thres 2 --photo_thres 0.6 "$@"

# -------------------------------------------------------------------------------------
# test on tanks & temples
#TANK_TESTING="/home/cjq/PycharmProjects/MVS/PatchmatchNet/data/TankandTemples/"
#python eval.py --scan_list lists/tanks/intermediate.txt --input_folder=$TANK_TESTING --output_folder=$TANK_TESTING \
#--checkpoint_path $CHECKPOINT_FILE --num_views 7 --image_max_dim 2048 --geo_mask_thres 5 --photo_thres 0.8 "$@"

# -------------------------------------------------------------------------------------
# test on your custom dataset
#CUSTOM_TESTING="/home/custom/"
#python eval.py --input_folder=$CUSTOM_TESTING --output_folder=$CUSTOM_TESTING --checkpoint_path $CHECKPOINT_FILE \
#--num_views 10 --image_max_dim 2048 --geo_mask_thres 5 --photo_thres 0.5 "$@"
