import copy
from typing import Optional, List
import numpy as np 
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from det3d.ops.deformDETR.modules import MSDeformAttn
from torch.nn.init import xavier_uniform_

class DeformableEncoderLayer(nn.Module):
    def __init__(self,
                 d_model = 128, cfg = None):
        super().__init__()

        # self attention
        d_ffn = cfg.feedforward_channel
        dropout = cfg.dropout
        activation = cfg.activation
        n_heads = cfg.num_heads
        n_points = cfg.enc_num_points
        n_levels = cfg.num_levels

        # self.channel_red = nn.Linear(d_model*2, d_model)
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, cur_src, src_all, pos, reference_points, 
                spatial_shape, level_start_index=None):
        # self attention
        # ref_src = torch.cat([cur_src, prev_src],2)
        # ref_src = self.channel_red(ref_src)
        # cur_pos = pos[-1]
        # import pdb; pdb.set_trace()
        if pos is not None:
            cur_pos = pos[-1]
            cur_src_with_pos = self.with_pos_embed(cur_src, cur_pos)
            src_all_with_pos = self.with_pos_embed(src_all, torch.cat(pos, 1))
        else:
            cur_src_with_pos = self.with_pos_embed(cur_src, pos)
            src_all_with_pos = self.with_pos_embed(src_all, pos)
        # input_level_start_index = torch.tensor([0]).long().cuda()
        src2 = self.self_attn(cur_src_with_pos, reference_points, src_all_with_pos, 
                              spatial_shape, level_start_index)
        # src2 = self.self_attn(self.with_pos_embed(ref_src, pos), reference_points,
        #                       prev_src, spatial_shape, input_level_start_index)
        src = cur_src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src

class DeformableEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        # import pdb; pdb.set_trace()
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]

        # H_, W_ = spatial_shape

        # ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, 
        #                     dtype=torch.float32, device=device), 
        #                     torch.linspace(0.5, W_ - 0.5, W_, 
        #                     dtype=torch.float32, device=device))
        # ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, 1] * H_)
        # ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, 0] * W_)
        # ref = torch.stack((ref_x, ref_y), -1)
        # reference_points_list.append(ref)
        # reference_points = torch.cat(reference_points_list, 1)
        # valid_ratios = valid_ratios.view(-1,1,2)
        # reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, cur_src, src_all, spatial_shapes, valid_ratios, level_start_index=None, pos=None):
        output = cur_src
        reference_points = self.get_reference_points(spatial_shapes[-1,:].view(1,-1), 
                                                     valid_ratios[:,-1,:].view(-1,1,2), 
                                                     device=cur_src.device)
        # spatial_shapes = spatial_shapes.view(1,-1)
        for _, layer in enumerate(self.layers):
            output = layer(output, src_all, pos, reference_points, spatial_shapes, level_start_index)

        return output

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
