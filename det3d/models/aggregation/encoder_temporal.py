import copy
from typing import Optional, List
import numpy as np 
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from det3d.ops.deformDETR.modules import MSDeformAttn, SeqMSDeformAttn
from torch.nn.init import xavier_uniform_

class Query_generator(nn.Module):
    def __init__(self, in_channels, seq_length, scale_hw):
        super(Query_generator, self).__init__()
        self.seq_len = seq_length
        self.func_unflatten = nn.Unflatten(dim=-1, unflattened_size=scale_hw)
        self.conv_encoding = nn.Conv2d(in_channels*2, in_channels, kernel_size=3, stride=1, padding=1)
        # Weight initialization
        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
    
    def forward(self, src_all):

        temporal_src = []
        target_bev_feats = self.func_unflatten(torch.transpose(src_all[0], 1,2))
        for idx in range(self.seq_len):
            support_bev_feats = self.func_unflatten(torch.transpose(src_all[idx], 1,2))
            encoded_bev_feats = torch.cat([target_bev_feats, support_bev_feats], 1)
            encoded_bev_feats = self.conv_encoding(encoded_bev_feats)
            encoded_src = encoded_bev_feats.flatten(2).transpose(1,2)
            temporal_src.append(encoded_src)
        
        assert len(temporal_src) == (self.seq_len)

        return temporal_src

class SeqDeformableEncoderLayer(nn.Module):
    def __init__(self,
                 d_model = 128, cfg = None,
                 feat_h=180, feat_w=180):
        super().__init__()

        # self attention
        d_ffn = cfg.feedforward_channel
        dropout = cfg.dropout
        activation = cfg.activation
        n_heads = cfg.num_heads
        n_points = cfg.enc_num_points
        n_levels = cfg.num_levels
        seq_len = cfg.seq_len

        self.query_generator = Query_generator(d_model, seq_len, (feat_h,feat_w))
        self.self_attn = SeqMSDeformAttn(d_model, n_levels, n_heads, n_points)
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

    def forward(self, src_all, pos, reference_points, 
                spatial_shape, level_start_index=None):

        query = self.query_generator(src_all)
        key = torch.cat(src_all, 1)
        src2 = self.self_attn(query, reference_points, key, 
                              spatial_shape, level_start_index)

        src = src_all[0] + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src

class SeqDeformableEncoder(nn.Module):
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

        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]

        return reference_points

    def forward(self, src_all, spatial_shapes, valid_ratios, level_start_index=None, pos=None):
        reference_points = self.get_reference_points(spatial_shapes[-1,:].view(1,-1), 
                                                     valid_ratios[:,-1,:].view(-1,1,2), 
                                                     device=src_all[0].device)
                                                     
        for idx, layer in enumerate(self.layers):
            output = layer(src_all, pos, reference_points, spatial_shapes, level_start_index)
            if idx == len(self.layers)-1:
                return output
            else:
                src_all = self.current_src_update(output, src_all)
    
    def current_src_update(self, cur_src, src_all):
        src_all[0] = cur_src
        return src_all

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
