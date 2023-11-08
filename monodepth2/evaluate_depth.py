from __future__ import absolute_import, division, print_function

import os

import cv2
import numpy as np
import torch
from monodepth2.options import MonodepthOptions

from layout_aware_monodepth.benchmarking.utils import run_eval_single_exp



splits_dir = os.path.join(os.path.dirname(__file__), "splits")

# Models which were trained with stereo supervision were trained with a nominal
# baseline of 0.1 units. The KITTI rig has a baseline of 54cm. Therefore,
# to convert our stereo predictions to real-world scale we multiply our depths by 5.4.
STEREO_SCALE_FACTOR = 5.4


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths"""
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25**2).mean()
    a3 = (thresh < 1.25**3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1"""
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def evaluate(opt):
    """Evaluates a pretrained model using a specified test set"""
    if opt.ds == "kitti":
        MIN_DEPTH = 1e-3
        MAX_DEPTH = 80
    elif opt.ds == "waymo":
        MIN_DEPTH = 1e-3
        MAX_DEPTH = 80
    elif opt.ds == "argoverse":
        MIN_DEPTH = 1e-3
        MAX_DEPTH = 180  # 200m is the max depth in the dataset, but the depths are distorted due to resizing

    assert (
        sum((opt.eval_mono, opt.eval_stereo)) == 1
    ), "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"

    if opt.ext_disp_to_eval is None:
        waymo_save_dir = (
            os.path.join(opt.load_weights_folder, "eval", "waymo")
            if opt.ds == "waymo"
            else None
        )
        assert opt.exp_name is not None
        _, pred_disps = run_eval_single_exp(
            opt.ds,
            opt.exp_name,
            waymo_save_dir=waymo_save_dir,
            root_dir="/mnt/wext/msc_studies/monodepth_project/artifacts",
            overwrite_preds=opt.overwrite_preds,
        )

    else:
        # Load predictions from file
        print("-> Loading predictions from {}".format(opt.ext_disp_to_eval))
        pred_disps = np.load(opt.ext_disp_to_eval)

        if opt.eval_eigen_to_benchmark:
            eigen_to_benchmark_ids = np.load(
                os.path.join(splits_dir, "benchmark", "eigen_to_benchmark_ids.npy")
            )

            pred_disps = pred_disps[eigen_to_benchmark_ids]

    if opt.save_pred_disps:
        output_path_depth = os.path.join(
            opt.load_weights_folder, "disps_{}_split.npy".format(opt.eval_split)
        )
        print("-> Saving predicted disparities to ", output_path_depth)
        np.save(output_path_depth, pred_disps)

    if opt.no_eval:
        print("-> Evaluation disabled. Done.")
        quit()

    elif opt.eval_split == "benchmark":
        save_dir = os.path.join(opt.load_weights_folder, "benchmark_predictions")
        print("-> Saving out benchmark predictions to {}".format(save_dir))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for idx in range(len(pred_disps)):
            disp_resized = cv2.resize(pred_disps[idx], (1216, 352))
            depth = STEREO_SCALE_FACTOR / disp_resized
            depth = np.clip(depth, 0, 80)
            depth = np.uint16(depth * 256)
            save_path = os.path.join(save_dir, "{:010d}.png".format(idx))
            cv2.imwrite(save_path, depth)

        print(
            "-> No ground truth is available for the KITTI benchmark, so not evaluating. Done."
        )
        quit()

    if opt.ds == "kitti":
        gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
        gt_depths = np.load(
            gt_path, fix_imports=True, encoding="latin1", allow_pickle=True
        )["data"]
    elif opt.ds == "waymo":
        gt_depths = np.load(
            os.path.join(splits_dir, opt.eval_split, "gt_depths_waymo.npy")
        )
    elif opt.ds == "argoverse":
        gt_depths = np.load(
            os.path.join(splits_dir, opt.eval_split, "gt_depths_argoverse.npy")
        )

    print("-> Evaluating")

    if opt.eval_stereo:
        print(
            "   Stereo evaluation - "
            "disabling median scaling, scaling by {}".format(STEREO_SCALE_FACTOR)
        )
        opt.disable_median_scaling = True
        opt.pred_depth_scale_factor = STEREO_SCALE_FACTOR
    else:
        print("   Mono evaluation - using median scaling")

    errors = []
    ratios = []

    assert len(pred_disps) == len(gt_depths)

    for i in range(len(pred_disps)):
        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = 1 / pred_disp

        if opt.save_pred_png:
            output_path_depth = os.path.join(
                opt.load_weights_folder,
                "eval",
                opt.ds,
                "depth",
                "{:010d}.png".format(i),
            )
            output_path_disp = output_path_depth.replace("depth/", "disp/")
            for p in [output_path_depth, output_path_disp]:
                os.makedirs(os.path.dirname(p), exist_ok=True)
            # if os.path.exists(output_path):
            #     print("-> {} already exists, aborting".format(output_path))
            #     break
            print("-> Saving predicted depth to ", output_path_depth)
            cv2.imwrite(output_path_depth, (pred_depth * 255).astype(np.uint16))
            cv2.imwrite(output_path_disp, (pred_disp * 255).astype(np.uint16))

        if opt.eval_split == "eigen":
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

            crop = np.array(
                [
                    0.40810811 * gt_height,
                    0.99189189 * gt_height,
                    0.03594771 * gt_width,
                    0.96405229 * gt_width,
                ]
            ).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0] : crop[1], crop[2] : crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)

        else:
            mask = gt_depth > 0

        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]

        pred_depth *= opt.pred_depth_scale_factor
        if not opt.disable_median_scaling:
            ratio = np.median(gt_depth) / np.median(pred_depth)
            ratios.append(ratio)
            pred_depth *= ratio

        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

        errors.append(compute_errors(gt_depth, pred_depth))

    if not opt.disable_median_scaling:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(
            " Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(
                med, np.std(ratios / med)
            )
        )

    mean_errors = np.array(errors).mean(0)

    header = "\n  " + ("{:>8} | " * 7).format(
        "abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"
    )
    res = ("&{: 8.3f}  " * 7).format(*mean_errors.tolist())
    print(header)
    print(res)
    with open(
        os.path.join(opt.load_weights_folder, f"eval/results_{opt.ds}.txt"), "w"
    ) as f:
        f.write(header + "\n")
        f.write(res + "\n")
    print("\n-> Done!")


def load_networks_for_eval(opt, model_name):
    encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
    decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

    encoder_dict = torch.load(encoder_path)

    if model_name == "monodepth2":
        encoder, depth_decoder = init_monodepth2_networks(opt)
    else:
        encoder, depth_decoder = init_sfmnext_networks(opt)

    model_dict = encoder.state_dict()
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
    depth_decoder.load_state_dict(torch.load(decoder_path))

    encoder.cuda()
    encoder.eval()
    depth_decoder.cuda()
    depth_decoder.eval()
    return encoder_dict, encoder, depth_decoder


def init_monodepth2_networks(opt):
    from monodepth2 import networks

    encoder = networks.ResnetEncoder(opt.num_layers, False)
    depth_decoder = networks.DepthDecoder(
        encoder.num_ch_enc, use_df_head=getattr(opt, "use_df_head", False)
    )
    return encoder, depth_decoder


def init_sfmnext_networks(opt):
    from sfmnext import networks

    if opt.backbone in ["resnet", "resnet_lite"]:
        encoder = networks.ResnetEncoderDecoder(
            num_layers=opt.num_layers,
            num_features=opt.num_features,
            model_dim=opt.model_dim,
        )
    elif opt.backbone == "resnet18_lite":
        encoder = networks.LiteResnetEncoderDecoder(model_dim=opt.model_dim)
    elif opt.backbone == "eff_b5":
        encoder = networks.BaseEncoder.build(
            num_features=opt.num_features, model_dim=opt.model_dim
        )
    else:
        encoder = networks.Unet(
            pretrained=(not opt.load_pretrained_model),
            backbone=opt.backbone,
            in_channels=3,
            num_classes=opt.model_dim,
            decoder_channels=opt.dec_channels,
        )

    if opt.backbone.endswith("_lite"):
        depth_decoder = networks.Lite_Depth_Decoder_QueryTr(
            in_channels=opt.model_dim,
            patch_size=opt.patch_size,
            dim_out=opt.dim_out,
            embedding_dim=opt.model_dim,
            query_nums=opt.query_nums,
            num_heads=4,
            min_val=opt.min_depth,
            max_val=opt.max_depth,
            use_df_head=False,
        )
    else:
        depth_decoder = networks.Depth_Decoder_QueryTr(
            in_channels=opt.model_dim,
            patch_size=opt.patch_size,
            dim_out=opt.dim_out,
            embedding_dim=opt.model_dim,
            query_nums=opt.query_nums,
            num_heads=4,
            min_val=opt.min_depth,
            max_val=opt.max_depth,
            use_df_head=getattr(opt, "use_df_head", False),
        )

    return encoder, depth_decoder


if __name__ == "__main__":
    cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)
    options = MonodepthOptions()
    evaluate(options.parse())
