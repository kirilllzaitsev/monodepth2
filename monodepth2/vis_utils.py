import numpy as np
import torch
import torchvision
from monodepth2.evaluate_depth import batch_post_process_disparity
from monodepth2.layers import disp_to_depth

from layout_aware_monodepth.vis_utils import plot_samples_and_preds


def plot_output_depths(depths, batch):
    fig = plot_samples_and_preds(
        batch,
        depths,
        with_colorbar=False,
        with_depth_diff=True,
        max_depth=80,
        with_img_depth_overlay=True,
        image=batch["color_aug", 0, 0].cpu(),
        depth=torchvision.transforms.Resize(depths.shape[2:], antialias=False)(
            batch["depth_gt"]
        ).cpu(),
    )
    return fig


def disp_to_depth_full(outputs):
    pred_disps = []
    pred_disps.append(outputs[("disp", 0)].cpu()[:, 0].detach())

    pred_disps = torch.concatenate(pred_disps)
    depths = []
    STEREO_SCALE_FACTOR = 5.4
    for idx in range(len(pred_disps)):
        # disp_resized = cv2.resize(pred_disps[idx], (1216, 352))
        disp_resized = pred_disps[idx]
        depth = STEREO_SCALE_FACTOR / disp_resized
        depth = torch.clip(depth, 0, 80)
        depths.append(depth)
    depths = torch.stack(depths, dim=0).unsqueeze(1)
    return pred_disps, depths


def disp_to_depth_full_v1(opt, data, depth_decoder, encoder):
    pred_disps = []
    with torch.no_grad():
        input_color = data[("color", 0, 0)].cuda()

        if opt.post_process:
            # Post-processed results require each image to have two forward passes
            input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)

        output = depth_decoder(encoder(input_color))

        pred_disp, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
        pred_disp = pred_disp.cpu()[:, 0].numpy()

        if opt.post_process:
            N = pred_disp.shape[0] // 2
            pred_disp = batch_post_process_disparity(
                pred_disp[:N], pred_disp[N:, :, ::-1]
            )

        pred_disps.append(pred_disp)

    pred_disps = np.concatenate(pred_disps)
    depths = []
    STEREO_SCALE_FACTOR = 5.4
    for idx in range(len(pred_disps)):
        # disp_resized = cv2.resize(pred_disps[idx], (1216, 352))
        disp_resized = pred_disps[idx]
        depth = STEREO_SCALE_FACTOR / disp_resized
        depth = np.clip(depth, 0, 80)
        depths.append(depth)
    depths = np.stack(depths, axis=0)
    return pred_disps, depths
