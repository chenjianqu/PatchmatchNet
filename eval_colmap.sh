#!/usr/bin/env bash

#CHECKPOINT_FILE="./checkpoints/params_000007.ckpt"
CHECKPOINT_FILE="/home/cjq/data/mvs/kaijin/params_000015.ckpt"

# -------------------------------------------------------------------------------------

CUSTOM_TESTING="/media/cjq/新加卷/datasets/220Dataset/22_GND_vslam/20230809_A12/20230805151650.846/dense/"

#python colmap_input.py --input_folder ${CUSTOM_TESTING}

# test on your custom dataset
#python eval_colmap.py --input_folder=$CUSTOM_TESTING --output_folder=$CUSTOM_TESTING --checkpoint_path $CHECKPOINT_FILE \
#--num_views 10 --image_max_dim 2048 --geo_mask_thres 3 --geo_depth_thres 0.05 --photo_thres 0.5 --crop_height 68 "$@"


python eval_colmap.py --input_folder=$CUSTOM_TESTING \
--mask_folder=${CUSTOM_TESTING}"/road_mask" \
--colmap_dense_folder=${CUSTOM_TESTING} \
--output_folder=$CUSTOM_TESTING \
--checkpoint_path $CHECKPOINT_FILE \
--num_views 5 \
--image_max_dim 640 \
--geo_mask_thres 3 \
--geo_depth_thres 0.1 \
--photo_thres 0.5 \
--output_type "both" \
--use_road_mask  "$@"

#python eval_colmap.py --input_folder=$CUSTOM_TESTING \
#--mask_folder=${CUSTOM_TESTING}"/road_mask" \
#--output_folder=$CUSTOM_TESTING --checkpoint_path $CHECKPOINT_FILE \
#--num_views 5 --image_max_dim 2048 --geo_mask_thres 3 --geo_depth_thres 0.05 --photo_thres 0.5 \
#--output_type "both" --use_road_mask  "$@"


#python eval_colmap.py --input_folder="/media/cjq/新加卷/datasets/220Dataset/22_GND_vslam/20230809_A12/20230805151650.846/dense/" \
#--output_folder="/media/cjq/新加卷/datasets/220Dataset/22_GND_vslam/20230809_A12/20230805151650.846/dense/" \
#--checkpoint_path "./checkpoints/params_000007.ckpt" \
#--num_views 10 --image_max_dim 2048 --geo_mask_thres 3 --photo_thres 0.5