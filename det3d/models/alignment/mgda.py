import time
import numpy as np
import math

import torch
from torchvision.ops.deform_conv import DeformConv2d

from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet
from torch.nn.modules.batchnorm import _BatchNorm

from det3d.torchie.cnn import constant_init, kaiming_init, xavier_init
from det3d.torchie.trainer import load_checkpoint
from det3d.models.utils import Empty, GroupNorm, Sequential
from det3d.models.utils import change_default_args

from .. import builder
from ..registry import ALIGNMENT
from ..utils import build_norm_layer
from .non_local_block import NONLocalBlock2D


@ALIGNMENT.register_module
class MGDA(nn.Module):
    def __init__(
        self,
        input_channel_list,
        encode_out_channel,
        target_channel,
        is_shared,
        mot_mode,
        is_down_sample,
        seq_len,
        sequence_mode,
        logger=None
    ):
        super(MGDA, self).__init__()
        self.input_channel_list = input_channel_list
        self.encode_out_channel = encode_out_channel
        self.target_channel = target_channel
        self.is_shared = is_shared
        self.mot_mode = mot_mode
        self.seq_len = seq_len
        self.sequence_mode = sequence_mode
        self.num_deform_location = 3*3

        encode_layer = []
        deconv_layer = []
        non_local_block = []
        for ms_idx, input_channel_size in enumerate(input_channel_list):
            if self.mot_mode == 'concat' or self.mot_mode == 'default':
                encode_layer.append(nn.Sequential(
                    nn.Conv2d(in_channels=input_channel_size*2, out_channels=64,
                            kernel_size=3, stride=2, padding=1, bias=True),
                    nn.LeakyReLU(inplace=True)
                ))
            else:
                encode_layer.append(nn.Sequential(
                    nn.Conv2d(in_channels=input_channel_size, out_channels=64,
                            kernel_size=3, stride=2, padding=1, bias=True),
                    nn.LeakyReLU(inplace=True)
                ))
            non_local_block.append(NONLocalBlock2D(in_channels=64))
            deconv_layer.append(nn.Sequential(
                nn.ConvTranspose2d(in_channels=64, out_channels=64, 
                                kernel_size=3, stride=2, padding=1,
                                output_padding=1, bias=True),
                nn.LeakyReLU(inplace=True)
            ))

        self.ms_encode_layer = nn.ModuleList(encode_layer)
        self.ms_non_local_block = nn.ModuleList(non_local_block)
        self.ms_deconv_layer = nn.ModuleList(deconv_layer)
        self.offset_final_conv = nn.Conv2d(
                in_channels=64, out_channels=self.num_deform_location*3,
                kernel_size=3, stride=1, padding=1,bias=True
            )
        self.sigmoid = nn.Sigmoid()

        self.dynamic_dcn = DeformConv2d(in_channels=self.target_channel,
                                        out_channels=self.target_channel,
                                        kernel_size=3, stride=1, padding=1, 
                                        bias=True)

        logger.info("Finish Motion-Aware Feature Alignment Initialization")

    def forward(self, x_all, ms_x_all):
        est_offset_all, est_weight_all = self.dynamic_offset_estimator(ms_x_all)
        align_x_all = [x_all[0]]
        for seq in range(self.seq_len-1):
            align_x_all.append(self.dynamic_dcn(x_all[seq+1],
                                                est_offset_all[seq],
                                                est_weight_all[seq]))
        
        return align_x_all
    
    def dynamic_offset_estimator(self, ms_x_all):

        cur_ms_x = ms_x_all[0]
        cur_ms_x.reverse()
        est_offset_all = []
        est_weight_all = []
        for seq in range(self.seq_len-1):
            prev_ms_x = ms_x_all[seq+1]
            prev_ms_x.reverse()
            mot_feat = self.extract_mot_feature(prev_ms_x, cur_ms_x, seq)

            est_result = self.offset_final_conv(mot_feat)
            est_offset = est_result[:,self.num_deform_location:,:,:]
            est_weight = self.sigmoid(est_result[:,:self.num_deform_location,:,:])
            est_offset_all.append(est_offset)
            est_weight_all.append(est_weight)
        
        return est_offset_all, est_weight_all
    
    def extract_mot_feature(self, prev_feat, cur_feat, seq):
        for ms_idx, (prev_x, cur_x) in enumerate(zip(prev_feat, cur_feat)):
            if self.mot_mode == 'concat':
                encode_x = self.ms_encode_layer[ms_idx](torch.cat([prev_x, cur_x], 1))
            elif self.mot_mode == 'sub':
                encode_x = self.ms_encode_layer[ms_idx](cur_x - prev_x)
            elif self.mot_mode == 'default':
                encode_x = self.ms_encode_layer[ms_idx](torch.cat([prev_x, cur_x-prev_x], 1))
            attn_x = self.ms_non_local_block[ms_idx](encode_x)
            mot_x = torch.add(encode_x, attn_x)

            if ms_idx == 0:
                up_x = self.ms_deconv_layer[ms_idx](mot_x)
            else:
                up_x = self.ms_deconv_layer[ms_idx](torch.add(mot_x, up_x))
        return up_x
