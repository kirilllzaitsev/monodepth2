#!/bin/bash

data_path=/kitti_raw_data

python train.py --exp_disabled --use_modelip_loss --modelip_loss_scale 0.1 --do_overfit --use_line_loss --model_name mono_model --png --split eigen_full --data_path ${data_path} --batch_size 1 --log_frequency  1 --num_workers 0 --num_epochs 200 --save_frequency 200 --scheduler_step_size 200 --learning_rate 5e-4