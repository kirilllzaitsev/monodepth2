# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from monodepth2.layers import Conv3x3, ConvBlock, upsample


class DepthDecoder(nn.Module):
    def __init__(
        self,
        num_ch_enc,
        scales=range(4),
        num_output_channels=1,
        use_skips=True,
        use_df_head=False,
    ):
        super().__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = "nearest"
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)
            if use_df_head:
                self.convs[("upconv_df", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)
            if use_df_head:
                self.convs[("upconv_df", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(
                self.num_ch_dec[s], self.num_output_channels
            )
            if use_df_head:
                self.convs[("dfconv", s)] = Conv3x3(self.num_ch_dec[s], 1)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()
        self.use_df_head = use_df_head

    def forward(self, input_features):
        self.outputs = {}

        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):
            if self.use_df_head:
                x_df = self.convs[("upconv_df", i, 0)](x)
                x_df = self.common_x(x_df, i, input_features)
                x_df = self.convs[("upconv_df", i, 1)](x_df)

            x = self.convs[("upconv", i, 0)](x)
            x = self.common_x(x, i, input_features)
            x = self.convs[("upconv", i, 1)](x)

            if i in self.scales:
                self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))
                if self.use_df_head:
                    self.outputs[("df", i)] = self.sigmoid(
                        self.convs[("dfconv", i)](x_df)
                    )

        return self.outputs

    def common_x(self, x, i, input_features):
        x = [upsample(x)]
        if self.use_skips and i > 0:
            x += [input_features[i - 1]]
        x = torch.cat(x, 1)
        return x
