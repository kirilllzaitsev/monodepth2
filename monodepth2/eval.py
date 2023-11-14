import argparse
import os
import sys

import cv2
from monodepth2.evaluate_depth import evaluate
from monodepth2.options import MonodepthOptions

from layout_aware_monodepth.benchmarking.utils import exp_name_to_meaning
from layout_aware_monodepth.benchmarking.utils import exp_names as experiments

parser = argparse.ArgumentParser()
parser.add_argument("--experiments", nargs="+", default=[])
parser.add_argument("--eval_ds", required=True, choices=["waymo", "kitti", "argoverse"])
parser.add_argument("--parse_preds", action="store_true")
args, _ = parser.parse_known_args()


root_dir = "/mnt/wext/msc_studies/monodepth_project/artifacts"

for exp_name in args.experiments:
    assert exp_name in exp_name_to_meaning
    exp_path = os.path.join(root_dir, exp_name)
    if exp_name in [k for k, v in exp_name_to_meaning.items() if "sfmnext" in v]:
        model_name = "sfmnext"
    else:
        model_name = "monodepth2"

    cmd = f"python evaluate_depth.py --model_name_general {model_name} --save_pred_png --ds {args.eval_ds} --png --eval_mono --load_weights_folder {exp_path} --data_path /mnt/wext/cv_data/kitti/kitti_raw_data --eval_split eigen --eval_out_dir {exp_path}/eval --exp_name {exp_name}"
    if args.parse_preds:
        cmd += f" --ext_disp_to_eval {exp_path}/eval/{args.eval_ds}/pred_disps.npy"

    cv2.setNumThreads(
        0
    )  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)
    options = MonodepthOptions()
    sys.argv = cmd.split(" ")[1:]
    opts = options.parse()
    evaluate(opts)

    return_code = os.system(cmd)
    if return_code != 0:
        exit("Error: os.system command failed")
