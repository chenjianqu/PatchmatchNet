#!/usr/bin/env bash

CUSTOM_TESTING="/media/cjq/新加卷/datasets/220Dataset/22_GND_vslam/20230809_A12/20230805151650.846/dense/"

python colmap_input.py --input_folder ${CUSTOM_TESTING} --crop_height 68

#python colmap_input.py --input_folder ${CUSTOM_TESTING}


