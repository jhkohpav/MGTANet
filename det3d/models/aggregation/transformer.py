import time
import numpy as np
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_

from .positional_embedding import PositionEmbeddingSine
from .encoder import DeformableEncoderLayer, DeformableEncoder
from det3d.ops.deformDETR.modules import MSDeformAttn

from .. import builder
from ..registry import AGGREGATION
from ..utils import build_norm_layer


@AGGREGATION.register_module
class Transformer(nn.Module):
    def __init__(
        self,
        src_in_channels,
        target_in_channels,
        feat_h,
        feat_w,
        seq_len,
        with_pos_emb,
        encoder_cfg,
        logger=None,
        **kwargs
    ):
        super(Transformer, self).__init__()
        self.src_input_channels     = src_in_channels
         # INPUT_TENSOR : Target tensor for deformable convolution
        self.encoder_cfg = encoder_cfg
        # assert type(self.src_input_channels) == list

        self.target_input_channels  = target_in_channels
        self.seq_length             = seq_len
        self.with_pos_emb           = with_pos_emb

        self.relu = nn.ReLU(inplace=True)
        self.channel_reduction = nn.Conv2d(self.target_input_channels , 
                                           self.src_input_channels, 3, 1, 1)
        
        self.pos_emb = PositionEmbeddingSine(num_pos_feats = int(self.src_input_channels/2))
        if self.with_pos_emb:
            self.embed = nn.ParameterList([nn.Parameter(torch.Tensor(self.src_input_channels)) 
                        for _ in range(self.seq_length)])
        encoder_layer = DeformableEncoderLayer(d_model = self.src_input_channels,
                                               cfg = self.encoder_cfg)
        self.encoder = DeformableEncoder(encoder_layer, self.encoder_cfg.num_layers)
        # self.encoder = self._get_clones(encoder, self.seq_length-1)

        self.unflatten = nn.Unflatten(2, (feat_h, feat_w))
        self.channel_extention = nn.Conv2d(self.src_input_channels, 
                                           self.target_input_channels, 3, 1, 1)

        self._reset_parameters()
        
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()

        if self.with_pos_emb:
            for i in range(self.seq_length):
                normal_(self.embed[i])
    
    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(mask[:, :, 0], 1)
        valid_W = torch.sum(mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio
    
    def _get_clones(self, module, N):
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
    
    def forward(self, aligned_x_all):
        #### prepare input for encoder ####
        # target_batch_dict = batch_dict_list[-1]
        cur_input_feats = aligned_x_all[0]
        cur_input_feats = self.relu(self.channel_reduction(cur_input_feats))
        batch_feats_list = [cur_input_feats]

        for i in range(self.seq_length-1):
            input_feats = aligned_x_all[i+1]
            input_feats = self.relu(self.channel_reduction(input_feats))
            batch_feats_list.append(input_feats)

        # import pdb; pdb.set_trace()
        # batch_feats_list = [self.relu(self.channel_reduction(x['aligned_features_2d'])) for i in range(self.seq_length)]
        pos, mask = self.pos_emb(batch_feats_list[0])
        if self.with_pos_emb:
            pos_flatten = pos.flatten(2).transpose(1,2)
            pos_emb_list = [pos_flatten + self.embed[i].cuda().view(1,1,-1) 
                        for i in range(self.seq_length)]

        b,c,h,w = batch_feats_list[0].shape
        spatial_shapes = [[h,w] for _ in range(self.seq_length)]

        src_list = [x.flatten(2).transpose(1,2) for x in batch_feats_list]
        # pos_emb_list = torch.cat(pos_emb_list, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_list[0].device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratio = [self.get_valid_ratio(mask).view(-1,1,2) 
                       for _ in range(self.seq_length)]
        valid_ratio = torch.cat(valid_ratio, 1)
        
        if self.with_pos_emb:
            memory = self.encoder(src_list[0], torch.cat(src_list,1), spatial_shapes,
                                valid_ratio, level_start_index, pos_emb_list)
        else:
            memory = self.encoder(src_list[0], torch.cat(src_list,1), spatial_shapes,
                                valid_ratio, level_start_index)
        memory = memory.transpose(1,2)
        encoded_feat = self.relu(self.channel_extention(self.unflatten(memory)))
        # for seq, encoder in enumerate(self.encoder):
        #     memory = encoder(src_list[-1], src_list[seq], spatial_shape,
        #                      valid_ratio, pos_emb_list[seq])
        #     memory = memory.transpose(1,2)
        #     encoded_feat = self.relu(self.channel_extention(self.unflatten(memory)))
        #     batch_dict_list[seq]['aligned_features_2d'] = encoded_feat
        
        return encoded_feat
