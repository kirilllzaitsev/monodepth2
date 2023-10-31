#!/bin/bash

data_path=/kitti_raw_data

# --use_line_loss --use_modelip_loss flags mean that both losses are used

python train.py --exp_disabled --exp_tags debug --use_line_loss --use_modelip_loss --use_df_rec_loss --filter_lines angle+length --model_name mono_model --png --split eigen_full --data_path ${data_path} --batch_size 8 --log_frequency 32 --num_workers 8