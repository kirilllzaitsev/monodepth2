from __future__ import absolute_import, division, print_function

import cv2
import matplotlib

matplotlib.use("Agg")

import argparse
import json
import os
import time

import comet_ml
import datasets
import matplotlib.pyplot as plt
import networks
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from IPython import embed
from kitti_utils import *
from layers import *
from monodepth2.line_losses import loss_function
from monodepth2.vis_utils import disp_to_depth_full, plot_output_depths
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from utils import *

from layout_aware_monodepth.line_utils import (
    filter_lines_by_angle,
    filter_lines_by_length,
    get_deeplsd_pred,
    load_deeplsd,
)
from layout_aware_monodepth.logging_utils import log_metric, log_params_to_exp
from layout_aware_monodepth.losses import LineLoss
from layout_aware_monodepth.metrics import calc_metrics
from layout_aware_monodepth.pipeline_utils import create_tracking_exp

# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
plt.ioff()


class Trainer:
    def __init__(self, options):
        self.opt = options
        self.exp = create_tracking_exp(options)
        log_params_to_exp(
            self.exp,
            vars(options),
            "options",
        )
        self.exp.add_tag("monodepth2")
        self.exp.add_tag("resnet18")
        self.exp.add_tag("overfit" if self.opt.do_overfit else "full")
        if self.opt.use_df_rec_loss:
            self.exp.add_tag("df_rec_loss")
        if self.opt.filter_lines is not None:
            self.exp.add_tag(f"filter_lines_{self.opt.filter_lines}")
        if self.opt.use_df_rec_loss:
            self.exp.add_tag(f"df_rec_loss")
        if "SLURM_JOB_ID" in os.environ:
            print("SLURM_JOB_ID", os.environ["SLURM_JOB_ID"])
        self.log_path = os.path.join(self.opt.log_dir, self.exp.name or "")
        os.makedirs(self.log_path, exist_ok=True)

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        if self.opt.use_modelip_loss:
            self.modelip_loss_scale = options.modelip_loss_scale

        self.models = {}
        self.parameters_to_train = []

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = (
            2 if self.opt.pose_model_input == "pairs" else self.num_input_frames
        )

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])

        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")

        self.models["encoder"] = networks.ResnetEncoder(
            self.opt.num_layers, self.opt.weights_init == "pretrained"
        )
        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())

        self.models["depth"] = networks.DepthDecoder(
            self.models["encoder"].num_ch_enc,
            self.opt.scales,
            use_df_head=self.opt.use_df_rec_loss,
        )
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())

        if self.use_pose_net:
            if self.opt.pose_model_type == "separate_resnet":
                self.models["pose_encoder"] = networks.ResnetEncoder(
                    self.opt.num_layers,
                    self.opt.weights_init == "pretrained",
                    num_input_images=self.num_pose_frames,
                )

                self.models["pose_encoder"].to(self.device)

                self.models["pose"] = networks.PoseDecoder(
                    self.models["pose_encoder"].num_ch_enc,
                    num_input_features=1,
                    num_frames_to_predict_for=2,
                )
                if self.opt.pretrained_pose_weights:
                    print("loading pretrained pose model")
                    self.load_pose_model()

            elif self.opt.pose_model_type == "shared":
                self.models["pose"] = networks.PoseDecoder(
                    self.models["encoder"].num_ch_enc, self.num_pose_frames
                )

            elif self.opt.pose_model_type == "posecnn":
                self.models["pose"] = networks.PoseCNN(
                    self.num_input_frames if self.opt.pose_model_input == "all" else 2
                )

            self.models["pose"].to(self.device)
            if not self.opt.pretrained_pose_weights:
                self.parameters_to_train += list(
                    self.models["pose_encoder"].parameters()
                )
                self.parameters_to_train += list(self.models["pose"].parameters())

        if self.opt.predictive_mask:
            assert (
                self.opt.disable_automasking
            ), "When using predictive_mask, please disable automasking with --disable_automasking"

            # Our implementation of the predictive masking baseline has the the same architecture
            # as our depth decoder. We predict a separate mask for each source frame.
            self.models["predictive_mask"] = networks.DepthDecoder(
                self.models["encoder"].num_ch_enc,
                self.opt.scales,
                num_output_channels=(len(self.opt.frame_ids) - 1),
            )
            self.models["predictive_mask"].to(self.device)
            self.parameters_to_train += list(
                self.models["predictive_mask"].parameters()
            )

        self.model_optimizer = optim.Adam(
            self.parameters_to_train, self.opt.learning_rate
        )
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer,
            self.opt.scheduler_step_size,
            0.1,
            verbose=not self.opt.do_overfit,
        )

        if self.opt.load_weights_folder is not None:
            self.load_model()

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        # data
        datasets_dict = {
            "kitti": datasets.KITTIRAWDataset,
            "kitti_odom": datasets.KITTIOdomDataset,
        }
        self.dataset = datasets_dict[self.opt.dataset]

        fpath = os.path.join(
            os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt"
        )

        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        img_ext = ".png" if self.opt.png else ".jpg"

        def exists(x):
            folder, index, _ = x.split(" ")
            return os.path.exists(
                os.path.join(
                    self.opt.data_path,
                    folder,
                    "proj_depth/groundtruth/image_02",
                    f"{int(index):010}.png",
                )
            ) and os.path.exists(
                os.path.join(
                    self.opt.data_path,
                    folder,
                    "velodyne_points/data",
                    f"{int(index):010}.bin",
                )
            )

        print(f"{len(train_filenames)=}")
        train_filenames = list(filter(exists, train_filenames))
        print(f"{len(train_filenames)=}")
        print(f"{len(val_filenames)=}")
        val_filenames = list(filter(exists, val_filenames))
        print(f"{len(val_filenames)=}")
        self.train_filenames = train_filenames
        self.val_filenames = val_filenames

        num_train_samples = len(train_filenames)
        self.num_total_steps = (
            num_train_samples // self.opt.batch_size * self.opt.num_epochs
        )

        train_dataset = self.dataset(
            self.opt.data_path,
            train_filenames,
            self.opt.height,
            self.opt.width,
            self.opt.frame_ids,
            4,
            is_train=True,
            img_ext=img_ext,
        )

        if self.opt.do_overfit:
            train_dataset = torch.utils.data.Subset(
                train_dataset, list(range(110, 110 + self.opt.batch_size))
            )
            val_dataset = train_dataset
        else:
            gen = torch.Generator()
            gen.manual_seed(0)
            random_idxs = torch.randperm(11_000, generator=gen)
            train_dataset = torch.utils.data.Subset(train_dataset, random_idxs)

            val_dataset = self.dataset(
                self.opt.data_path,
                val_filenames,
                self.opt.height,
                self.opt.width,
                self.opt.frame_ids,
                4,
                is_train=False,
                img_ext=img_ext,
            )

        self.train_loader = DataLoader(
            train_dataset,
            self.opt.batch_size,
            True,
            num_workers=self.opt.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        self.val_loader = DataLoader(
            val_dataset,
            self.opt.batch_size,
            False,
            num_workers=self.opt.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        self.val_iter = iter(self.val_loader)

        self.benchmark_batch = next(iter(self.val_loader))

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2**scale)
            w = self.opt.width // (2**scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel",
            "de/sq_rel",
            "de/rms",
            "de/log_rms",
            "da/a1",
            "da/a2",
            "da/a3",
        ]

        print("Using split:\n  ", self.opt.split)
        print(
            "There are {:d} training items and {:d} validation items\n".format(
                len(train_dataset), len(val_dataset)
            )
        )

        self.save_opts()

        if options.use_modelip_loss:
            self.exp.add_tag("modelip_loss")
        if options.use_line_loss:
            self.exp.add_tag("line_loss")
            self.line_loss = LineLoss()
            self.line_loss_scale = options.line_loss_scale
        self.exp.add_tags(options.exp_tags)

        if options.use_line_loss or options.use_modelip_loss:
            self.dlsd = load_deeplsd().to(self.device)

    def set_train(self):
        """Convert all models to training mode"""
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode"""
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline"""
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()
            # if (self.epoch + 1) % self.opt.save_frequency == 0:
            if not self.opt.do_overfit:
                self.save_model(self.epoch)

    def run_epoch(self):
        """Run a single epoch of training and validation"""

        print("Training")
        self.set_train()

        for batch_idx, inputs in enumerate(self.train_loader):
            before_op_time = time.time()

            outputs, losses = self.process_batch(inputs)

            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()

            duration = time.time() - before_op_time

            # log less frequently after the first 500 steps to save time & disk space
            if self.opt.do_overfit:
                early_phase = self.epoch % self.opt.log_frequency == 0
            else:
                early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < (
                    self.opt.log_frequency * 10
                )
            late_phase = self.step % (self.opt.log_frequency * 10) == 0

            if early_phase or late_phase:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)

                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)

                self.log("train", inputs, outputs, losses)
                self.val()

            self.step += 1

        self.model_lr_scheduler.step()

        # Enforce the minimum learning rate
        for param_group in self.model_optimizer.param_groups:
            param_group["lr"] = max(param_group["lr"], 5e-6)

    # def calc_metrics(self, outputs, batch):
    #     _, depths = disp_to_depth_full(outputs, self.opt)
    #     gt = batch["depth_gt"].cpu()
    #     depths = torchvision.transforms.Resize(gt.shape[2:], antialias=False)(
    #         depths
    #     ).cpu()
    #     assert depths.max() > 1 and gt.max() > 1

    #     return calc_metrics(
    #         gt,
    #         depths,
    #     )

    def log_metric(self, exp, metrics: dict, step: int, prefix: str = None):
        for k, v in metrics.items():
            if prefix is not None:
                k = f"{prefix}/{k}"

            exp.log_metric(k, v, step=step)

    def do_benchmark(self):
        outputs, _ = self.process_batch(self.benchmark_batch)
        # _, depths = disp_to_depth_full(outputs, self.opt)
        depths = outputs[("depth", 0, 0)].cpu()

        fig = plot_output_depths(depths, self.benchmark_batch)
        name = "preds/sample"
        self.exp.log_figure(
            name,
            fig,
            step=self.step,
        )
        plt.close()

        if self.opt.use_line_loss or self.opt.use_modelip_loss:
            fig = self.plot_lines()

            name = "preds/lines"
            self.exp.log_figure(
                name,
                fig,
                step=self.step,
            )

            plt.close()

        if self.opt.use_line_reproj_loss:
            fig = self.plot_reproj_lines()
            name = "preds/reproj_lines"
            self.exp.log_figure(
                name,
                fig,
                step=self.step,
            )
            plt.close()

    def plot_reproj_lines(self):
        outputs, _ = self.process_batch(self.benchmark_batch)
        reproj_res = self.compute_line_reproj_loss(self.benchmark_batch, outputs)
        
        batch_size = self.opt.batch_size
        ncols = 5
        fig, axs = plt.subplots(
            batch_size,
            ncols,
            figsize=(max(batch_size * 5, 10), ncols * 5),
        )
        for i, closest_res in enumerate(reproj_res['res']['closest_res_batch']):
            if batch_size == 1:
                ax_0 = axs[0]
                ax_1 = axs[1]
                ax_2 = axs[2]
                ax_3 = axs[3]
                ax_4 = axs[4]
            else:
                ax_0 = axs[i, 0]
                ax_1 = axs[i, 1]
                ax_2 = axs[i, 2]
                ax_3 = axs[i, 3]
                ax_4 = axs[i, 4]

            paired_lines, avg_distances = (
                closest_res["paired_lines"],
                closest_res["avg_distances"],
            )
            reproj_paired = plot_line_pairs(
                paired_lines["reproj"],
                paired_lines["true"],
                take_n=30,
                hw=(self.opt.height, self.opt.width),
                avg_distances=avg_distances,
                font_scale=2,
                no_text=True,
            )
            # image, lines_t, lines_t_1, reprojection
            image = self.benchmark_batch["color_aug", 0, 0][i].cpu()
            hw = (self.opt.height, self.opt.width)
            lines_t = plot_lines(lines=reproj_res['res']["lines_t"][i], hw=hw)
            lines_t_1 = plot_lines(lines=reproj_res['res']["lines_t_1"][i], hw=hw)
            ax_0.imshow(image.permute(1, 2, 0))
            ax_1.imshow(lines_t)
            ax_2.imshow(lines_t_1)
            ax_3.imshow(plot_lines(reproj_res['res']['repr_lines'][i], hw=hw))
            ax_4.imshow(reproj_paired)

            ax_0.axis("off")
            ax_1.axis("off")
            ax_2.axis("off")
            ax_3.axis("off")
            ax_4.axis("off")
            
            if i == 0:
                ax_0.set_title("image")
                ax_1.set_title("lines_t")
                ax_2.set_title("lines_t_1")
                ax_3.set_title("reproj")
                ax_4.set_title("reproj (top-N distant)")
        return fig

    def plot_lines(self):
        x = self.benchmark_batch[("color_aug", 0, 0)].to(self.device)
        ls, df = self.get_lines(x, include_df=True)
        x_other_side = self.benchmark_batch[("color_other_side", 0, 0)].to(self.device)
        ls2, df2 = self.get_lines(x_other_side, include_df=True)

        batch_size = len(x)
        ncols = 2 + 2
        fig, axs = plt.subplots(
            batch_size,
            ncols,
            figsize=(max(batch_size * 5, 10), ncols * 5),
        )
        for i in range(batch_size):
            if batch_size == 1:
                ax_0 = axs[0]
                ax_1 = axs[1]
                ax_2 = axs[2]
                ax_3 = axs[3]
            else:
                ax_0 = axs[i, 0]
                ax_1 = axs[i, 1]
                ax_2 = axs[i, 2]
                ax_3 = axs[i, 3]
            concat1 = np.zeros((self.opt.height, self.opt.width, 1))
            for line in ls[i].astype("int"):
                concat1 = cv2.line(
                    concat1, tuple(line[0]), tuple(line[1]), (1, 1, 1), 2
                )
            concat_other_side = np.zeros((self.opt.height, self.opt.width, 1))
            for line in ls2[i].astype("int"):
                concat_other_side = cv2.line(
                    concat_other_side, tuple(line[0]), tuple(line[1]), (1, 1, 1), 2
                )
            ax_0.imshow(concat1.squeeze())
            ax_1.imshow(df[i].cpu().squeeze())
            ax_2.imshow(concat_other_side.squeeze())
            ax_3.imshow(df2[i].cpu().squeeze())

            ax_0.axis("off")
            ax_1.axis("off")
            ax_2.axis("off")
            ax_3.axis("off")
            if i == 0:
                ax_0.set_title("lines")
                ax_1.set_title("df")
                ax_2.set_title("lines other side")
                ax_3.set_title("df other side")
        return fig

    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses"""
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        if self.opt.pose_model_type == "shared":
            # If we are using a shared encoder for both depth and pose (as advocated
            # in monodepthv1), then all images are fed separately through the depth encoder.
            all_color_aug = torch.cat(
                [inputs[("color_aug", i, 0)] for i in self.opt.frame_ids]
            )
            all_features = self.models["encoder"](all_color_aug)
            all_features = [torch.split(f, self.opt.batch_size) for f in all_features]

            features = {}
            for i, k in enumerate(self.opt.frame_ids):
                features[k] = [f[i] for f in all_features]

            outputs = self.models["depth"](features[0])
        else:
            # Otherwise, we only feed the image with frame_id 0 through the depth encoder
            features = self.models["encoder"](inputs["color_aug", 0, 0])
            outputs = self.models["depth"](features)

        if self.opt.predictive_mask:
            outputs["predictive_mask"] = self.models["predictive_mask"](features)

        if self.use_pose_net:
            outputs.update(self.predict_poses(inputs, features))

        self.generate_images_pred(inputs, outputs)
        losses = self.compute_losses(inputs, outputs)

        return outputs, losses

    def predict_poses(self, inputs, features):
        """Predict poses between input frames for monocular sequences."""
        outputs = {}
        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # select what features the pose network takes as input
            if self.opt.pose_model_type == "shared":
                pose_feats = {f_i: features[f_i] for f_i in self.opt.frame_ids}
            else:
                pose_feats = {
                    f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids
                }

            for f_i in self.opt.frame_ids[1:]:
                if f_i != "s":
                    # To maintain ordering we always pass frames in temporal order
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    if self.opt.pose_model_type == "separate_resnet":
                        pose_inputs = [
                            self.models["pose_encoder"](torch.cat(pose_inputs, 1))
                        ]
                    elif self.opt.pose_model_type == "posecnn":
                        pose_inputs = torch.cat(pose_inputs, 1)

                    axisangle, translation = self.models["pose"](pose_inputs)
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation

                    # Invert the matrix if the frame id is negative
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=(f_i < 0)
                    )

        else:
            # Here we input all frames to the pose net (and predict all poses) together
            if self.opt.pose_model_type in ["separate_resnet", "posecnn"]:
                pose_inputs = torch.cat(
                    [
                        inputs[("color_aug", i, 0)]
                        for i in self.opt.frame_ids
                        if i != "s"
                    ],
                    1,
                )

                if self.opt.pose_model_type == "separate_resnet":
                    pose_inputs = [self.models["pose_encoder"](pose_inputs)]

            elif self.opt.pose_model_type == "shared":
                pose_inputs = [features[i] for i in self.opt.frame_ids if i != "s"]

            axisangle, translation = self.models["pose"](pose_inputs)

            for i, f_i in enumerate(self.opt.frame_ids[1:]):
                if f_i != "s":
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, i], translation[:, i]
                    )

        return outputs

    def val(self):
        """Validate the model on a single minibatch"""
        self.set_eval()
        try:
            inputs = next(self.val_iter)
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = next(self.val_iter)

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)

            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)

            self.log("val", inputs, outputs, losses)

            self.do_benchmark()
            # outputs, _ = self.process_batch(self.benchmark_batch)
            # the metrics below is calculated in compute_depth_losses and put in losses
            # metrics = self.calc_metrics(outputs, self.benchmark_batch)
            # self.log_metric(self.exp, metrics, self.step, prefix="benchmark")
            self.log_metric(self.exp, losses, self.step, prefix="benchmark")

            del inputs, outputs, losses

        self.set_train()

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(
                    disp,
                    [self.opt.height, self.opt.width],
                    mode="bilinear",
                    align_corners=False,
                )
                source_scale = 0

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):
                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]

                # from the authors of https://arxiv.org/abs/1712.00175
                if self.opt.pose_model_type == "posecnn":
                    axisangle = outputs[("axisangle", 0, frame_id)]
                    translation = outputs[("translation", 0, frame_id)]

                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

                    T = transformation_from_parameters(
                        axisangle[:, 0],
                        translation[:, 0] * mean_inv_depth[:, 0],
                        frame_id < 0,
                    )

                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)]
                )
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T
                )

                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border",
                )

                if not self.opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = inputs[
                        ("color", frame_id, source_scale)
                    ]

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images"""
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch"""
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]

            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))

            reprojection_losses = torch.cat(reprojection_losses, 1)

            if not self.opt.disable_automasking:
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    pred = inputs[("color", frame_id, source_scale)]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target)
                    )

                identity_reprojection_losses = torch.cat(
                    identity_reprojection_losses, 1
                )

                if self.opt.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(
                        1, keepdim=True
                    )
                else:
                    # save both images, and do min all at once below
                    identity_reprojection_loss = identity_reprojection_losses

            elif self.opt.predictive_mask:
                # use the predicted mask
                mask = outputs["predictive_mask"]["disp", scale]
                if not self.opt.v1_multiscale:
                    mask = F.interpolate(
                        mask,
                        [self.opt.height, self.opt.width],
                        mode="bilinear",
                        align_corners=False,
                    )

                reprojection_losses *= mask

                # add a loss pushing mask to 1 (using nn.BCELoss for stability)
                weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).cuda())
                loss += weighting_loss.mean()

            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses

            if not self.opt.disable_automasking:
                # add random numbers to break ties
                identity_reprojection_loss += (
                    torch.randn(identity_reprojection_loss.shape, device=self.device)
                    * 0.00001
                )

                combined = torch.cat(
                    (identity_reprojection_loss, reprojection_loss), dim=1
                )
            else:
                combined = reprojection_loss

            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)

            if not self.opt.disable_automasking:
                outputs["identity_selection/{}".format(scale)] = (
                    idxs > identity_reprojection_loss.shape[1] - 1
                ).float()

            loss += to_optimise.mean()

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.opt.disparity_smoothness * smooth_loss / (2**scale)

            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales

        # the rest gets computed only for the highest scale
        if self.opt.use_line_loss:
            line_loss = self.compute_line_loss(inputs, outputs)
            total_loss += self.line_loss_scale * line_loss
            losses["line_loss"] = line_loss

        if self.opt.use_modelip_loss:
            modelip_loss = self.compute_modelip_loss(inputs, outputs)
            total_loss += self.opt.modelip_loss_scale * modelip_loss["modelip_loss"]
            for k, v in modelip_loss.items():
                losses[k] = v

        if self.opt.use_line_reproj_loss:
            line_reproj_res = self.compute_line_reproj_loss(inputs, outputs)
            total_loss += self.opt.line_reproj_loss_scale * line_reproj_res["line_reproj_loss"]
            for k, v in line_reproj_res.items():
                if "_loss" in k:
                    losses[k] = v

        losses["loss"] = total_loss
        return losses

    def compute_line_reproj_loss(self, batch, out):
        y_pred = out[("depth", 0, 0)].to(self.device)
        K = torch.zeros((len(y_pred), 4, 4)).to(self.device)
        K[...,:3, :3] = batch["intrinsics"]
        K[..., 3, 3] = 1
        Ki = torch.pinverse(K)
        # backproj only onto the next frame
        pose = out[("cam_T_cam", 0, 1)]

        x_t = batch[("color_aug", 0, 0)].to(self.device)
        lines_t = self.get_lines(x_t)
        x_t_1 = batch[("color_aug", 1, 0)].to(self.device)
        lines_t_1 = self.get_lines(x_t_1)

        repr_lines = reproject_lines_batch(lines_t, pose, K, Ki, y_pred)
        closest_res_batch = find_closest_lines_to_src_batch(repr_lines, lines_t_1)

        line_dist_loss = torch.tensor(0.0).to(self.device)
        line_orient_loss = torch.tensor(0.0).to(self.device)
        for closest_res in closest_res_batch:
            line_dist_loss += line_distance_loss((closest_res["avg_distances"]))
            line_orient_loss += line_orientation_loss(
                closest_res["paired_lines"]["reproj"],
                closest_res["paired_lines"]["true"],
            )
        # 1 line_dist_loss ~ 5-15
        # 1 line_orient_loss ~ 0.05-0.1
        loss = 0.2*line_dist_loss + 10*line_orient_loss
        return {
            "line_reproj_loss": loss,
            "line_dist_loss": line_dist_loss,
            "line_orient_loss": line_orient_loss,
            "res": {
                "lines_t": lines_t,
                "lines_t_1": lines_t_1,
                "repr_lines": repr_lines,
                "closest_res_batch": closest_res_batch,
            }
        }

    def compute_modelip_loss(self, batch, out):
        # _, y_pred = disp_to_depth_full(out, self.opt)
        y_pred = out[("depth", 0, 0)]
        y_pred = y_pred.to(self.device)
        K = batch["intrinsics"].to(self.device)
        Ki = torch.inverse(K)
        args = argparse.Namespace(
            **{
                "calibration": 0,
                "w1": 0.1,
                "w2": 1.0,
                "w3": 0.1,
                "w4": 0.1,
                "w5": 0.1,
                "w6": 0.001,
                "w7": 0.1,
                "w8": 0.1,
                "max_depth": 80,
            }
        )
        loss = torch.tensor(0.0).to(self.device)
        x = batch[("color_aug", 0, 0)].to(self.device)
        ls, df = self.get_lines(x, include_df=True)

        # Build the transformation matrices
        P1 = batch["P1"].to(self.device)
        P2 = batch["P2"].to(self.device)
        hom = (
            torch.tensor([[[0.0, 0.0, 0.0, 1.0]]]).repeat(len(ls), 1, 1).to(self.device)
        )
        P1 = torch.cat([P1, hom], dim=1)
        P2 = torch.cat([P2, hom], dim=1)
        Q = torch.matmul(P2, torch.inverse(P1))
        # df = batch["df"].to(self.device)
        # df2 = batch["df_2"].to(self.device)

        # df = torchvision.transforms.Resize(y_pred.shape[2:], antialias=False)(df)
        # df2 = torchvision.transforms.Resize(y_pred.shape[2:], antialias=False)(df2)
        # side=batch["side"]
        # img_path=batch["img_path"]
        x_other_side = batch[("color_other_side", 0, 0)].to(self.device)
        _, df2 = self.get_lines(x_other_side, include_df=True)

        def prepare_df(df):
            if len(df.shape) == 3:
                df = df.unsqueeze(1)
            return df

        df = prepare_df(df)
        df2 = prepare_df(df2)

        df_pred = None
        if self.opt.use_df_rec_loss:
            df_pred = out[("df", 0)]
            df_pred = prepare_df(df_pred)

        y_true = torchvision.transforms.Resize(y_pred.shape[2:], antialias=False)(
            batch["depth_gt"]
        )

        loss = loss_function(
            y_true=y_true,
            y_pred=y_pred,
            ls=ls,
            Ki=Ki,
            args=args,
            Q=Q,
            df_true=df / 5.0,
            df_pred=df_pred,
            include_df_rec_loss=self.opt.use_df_rec_loss,
            df1=df,
            df2=df2,
            do_ssl=True,
            include_df_proj_loss=True,
        )
        return loss

    def compute_line_loss(self, batch, out):
        x = batch[("color_aug", 0, 0)].to(self.device)
        lines = self.get_lines(x)
        # _, depths = disp_to_depth_full(out, self.opt)
        depths = out[("depth", 0, 0)]
        loss_line = self.line_loss(depths, lines)
        return loss_line

    def get_lines(self, x, include_df=False):
        line_res = get_deeplsd_pred(self.dlsd, x)
        lines = line_res["lines"]
        if self.opt.filter_lines is not None:
            filtered_lines = lines
            if "length" in self.opt.filter_lines:
                filtered_lines = filter_lines_by_length(
                    lines, use_min_length=True, min_length=20
                )
            if "angle" in self.opt.filter_lines:
                filtered_lines = filter_lines_by_angle(
                    filtered_lines, low_thresh=np.pi / 15, high_thresh=np.pi / 2.25
                )
            if len(filtered_lines) == 0:
                print("no lines left after filtering")
            lines = filtered_lines
        if include_df:
            return lines, line_res["df"]
        return lines

    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(
            F.interpolate(
                depth_pred, [375, 1242], mode="bilinear", align_corners=False
            ),
            1e-3,
            80,
        )
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = depth_gt > 0

        # garg/eigen crop
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, 153:371, 44:1197] = 1
        mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal"""
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            (self.num_total_steps / self.step - 1.0) * time_sofar
            if self.step > 0
            else 0
        )
        print_string = (
            "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}"
            + " | loss: {:.5f} | time elapsed: {} | time left: {}"
        )
        print(
            print_string.format(
                self.epoch,
                batch_idx,
                samples_per_sec,
                loss,
                sec_to_hm_str(time_sofar),
                sec_to_hm_str(training_time_left),
            )
        )

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file"""
        writer = self.writers[mode]
        for l, v in losses.items():
            if "/" in l:
                l = l.replace("/", "/scale_")
            writer.add_scalar("{}".format(l), v, self.step)
            self.exp.log_metric(f"{mode}/{l}", v, step=self.step)

        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            for s in self.opt.scales:
                for frame_id in self.opt.frame_ids:
                    writer.add_image(
                        "color_{}_{}/{}".format(frame_id, s, j),
                        inputs[("color", frame_id, s)][j].data,
                        self.step,
                    )
                    self.exp.log_image(
                        inputs[("color", frame_id, s)][j].data.permute(1, 2, 0).cpu(),
                        f"{mode}/color_{frame_id}_{s}/{j}",
                        step=self.step,
                    )
                    if s == 0 and frame_id != 0:
                        writer.add_image(
                            "color_pred_{}_{}/{}".format(frame_id, s, j),
                            outputs[("color", frame_id, s)][j].data,
                            self.step,
                        )
                        self.exp.log_image(
                            outputs[("color", frame_id, s)][j]
                            .data.permute(1, 2, 0)
                            .cpu(),
                            f"{mode}/color_pred_{frame_id}_{s}/{j}",
                            step=self.step,
                        )

                writer.add_image(
                    "disp_{}/{}".format(s, j),
                    normalize_image(outputs[("disp", s)][j]),
                    self.step,
                )
                self.exp.log_image(
                    normalize_image(outputs[("disp", s)][j]).permute(1, 2, 0).cpu(),
                    f"{mode}/disp_{s}/{j}",
                    step=self.step,
                )

                if self.opt.predictive_mask:
                    for f_idx, frame_id in enumerate(self.opt.frame_ids[1:]):
                        writer.add_image(
                            "predictive_mask_{}_{}/{}".format(frame_id, s, j),
                            outputs["predictive_mask"][("disp", s)][j, f_idx][
                                None, ...
                            ],
                            self.step,
                        )

                elif not self.opt.disable_automasking:
                    writer.add_image(
                        "automask_{}/{}".format(s, j),
                        outputs["identity_selection/{}".format(s)][j][None, ...],
                        self.step,
                    )

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with"""
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, "opt.json"), "w") as f:
            json.dump(to_save, f, indent=2)

    def save_model(self, epoch):
        """Save model weights to disk"""
        save_folder = os.path.join(
            self.log_path, "models", "weights_{}".format(self.epoch)
        )
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == "encoder":
                # save the sizes - these are needed at prediction time
                to_save["height"] = self.opt.height
                to_save["width"] = self.opt.width
                to_save["use_stereo"] = self.opt.use_stereo
            torch.save(to_save, save_path)
            self.exp.log_model(f"{model_name}_{epoch}", save_path, overwrite=False)
            print("Saved {} weights at {}".format(model_name, save_path))

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

    def load_model(self):
        """Load model(s) from disk"""
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(
            self.opt.load_weights_folder
        ), "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {
                k: v for k, v in pretrained_dict.items() if k in model_dict
            }
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")

    def load_pose_model(self, models, weights_dir):
        for n in ["pose", "pose_encoder"]:
            print("Loading {} weights...".format(n))
            path = os.path.join(weights_dir, "{}.pth".format(n))
            model_dict = models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {
                k: v for k, v in pretrained_dict.items() if k in model_dict
            }
            model_dict.update(pretrained_dict)
            models[n].load_state_dict(model_dict)


def load_pose_resnets(opt, device):
    models = {}
    models["pose_encoder"] = networks.ResnetEncoder(
        opt.num_layers,
        opt.weights_init == "pretrained",
        num_input_images=2,
    )

    models["pose"] = networks.PoseDecoder(
        models["pose_encoder"].num_ch_enc,
        num_input_features=1,
        num_frames_to_predict_for=2,
    )
    models["pose_encoder"].to(device)
    models["pose"].to(device)

    return models
