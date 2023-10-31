#!/bin/bash

# REQUIRES having predictions exported to /tmp/artifacts/${exp_name}/eval/${eval_ds}/pred_disps.npy

data_path=/kitti_raw_data

exp_name=stiff_wolverine_3619  # name of an experiment on comet that contains trained model artifacts ("Assets"->"models/"), https://www.comet.com/kirilllzaitsev/layout-aware-monodepth/59e313a8f5de4d37899eaa988655bfaa
exp_path=/tmp/artifacts/${exp_name}
eval_ds=kitti
# eval_ds=waymo
# eval_ds=argoverse

python evaluate_depth.py --ds ${eval_ds} --ext_disp_to_eval ${exp_path}/eval/${eval_ds}/pred_disps.npy --png --eval_mono --load_weights_folder ${exp_path} --data_path ${data_path} --eval_split eigen --eval_out_dir ${exp_path}/eval