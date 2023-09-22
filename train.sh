#!/usr/bin/env bash

MVS_TRAINING_RAW="/media/cjq/新加卷/datasets/OpenSourceDatasets/training/dtu"
MVS_TRAINING="/media/cjq/新加卷/datasets/OpenSourceDatasets/training/dtu_PatchmatchNet"

# 将dtu数据集进行预处理
#python ./convert_dtu_dataset.py --input_folder $MVS_TRAINING_RAW --output_folder ${MVS_TRAINING} --scan_list ../lists/dtu/all.txt



# Train on converted DTU training set
python train.py --batch_size 3 --epochs 8 --num_light_idx 7 --input_folder=$MVS_TRAINING --output_folder=$MVS_TRAINING \
--train_list=lists/dtu/train.txt --test_list=lists/dtu/val.txt --image_max_dim 400 "$@"



# Legacy train on DTU's training set
#python train_dtu.py --batch_size 4 --epochs 8 --trainpath=$MVS_TRAINING --trainlist lists/dtu/train.txt \
#--vallist lists/dtu/val.txt --logdir ./checkpoints "$@"
