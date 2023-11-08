import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--experiments", nargs="+", default=[])
parser.add_argument("--eval_ds", required=True, choices=["waymo", "kitti", "argoverse"])
parser.add_argument("--parse_preds", action="store_true")
args, _ = parser.parse_known_args()

experiments = [
    "estimated_puffin_8334",
    "sufficient_elbow_5310",
    "obvious_limo_8086",
    "good_tortoise_9774",
    "environmental_pie_3115",
]
exp_name_to_meaning = {
    "big_cuckoo_8361": "monodepth2 modelip loss w/o df head",
    "grumpy_board_7469": "monodepth2 line loss",
    "estimated_puffin_8334": "monodepth2 vanilla",
    "stiff_wolverine_3619": "monodepth2 modelip loss w/ df head",
    "blush_herring_9327": "monodepth2 line+modelip w/ line filtering",
    "sufficient_elbow_5310": "sfmnext vanilla",
    "obvious_limo_8086": "sfmnext modelip loss w/ DF reconstruction",
    "good_tortoise_9774": "monodepth2 fixed modelip loss w/ df head",
    "environmental_pie_3115": "monodepth2 fixed modelip loss w/o df head",
}


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

    return_code = os.system(cmd)
    if return_code != 0:
        exit("Error: os.system command failed")
