import torch
from torch import nn
from torch.nn import functional as F

from ..registry import READERS



@READERS.register_module
class VoxelFeatureExtractorV3(nn.Module):
    def __init__(
        self, num_input_features=4, norm_cfg=None, name="VoxelFeatureExtractorV3"
    ):
        super(VoxelFeatureExtractorV3, self).__init__()
        self.name = name
        self.num_input_features = num_input_features

    def forward(self, features, num_voxels, coors=None):
        assert self.num_input_features == features.shape[-1]

        points_mean = features[:, :, : self.num_input_features].sum(
            dim=1, keepdim=False
        ) / num_voxels.type_as(features).view(-1, 1)

        return points_mean.contiguous()

@READERS.register_module
class SM_VFE(nn.Module):
    def __init__(
        self, num_input_features=4, norm_cfg=None, name="SM_VFE",
        nsweeps = 10
    ):
        super(SM_VFE, self).__init__()
        self.name = name
        self.num_input_features = num_input_features
        self.nsweeps = nsweeps
        self.layer1 = nn.Sequential(
            nn.Linear(self.num_input_features, 16),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(16, 16),
            nn.Sigmoid()
        )
        self.layer3 = nn.Sequential(
            nn.Linear(16*(self.nsweeps-1), 16),
            nn.ReLU(inplace=True)
        )

    def forward(self, features, num_voxels, coors=None):
        assert self.num_input_features == features.shape[-1]

        temp_features = self.seq_points_mean(features, num_voxels)
        points_mean = features[:, :, : self.num_input_features].sum(
            dim=1, keepdim=False
        ) / num_voxels.type_as(features).view(-1, 1)

        seq_voxel_feats = torch.cat([temp_features, points_mean.contiguous()], 1)

        return seq_voxel_feats

    def seq_points_mean(self, features, num_voxels):
        split_features = []
        split_num_voxels = []
        for i in range(self.nsweeps):
            if i == 0:
                mask = (features[:,:,-1]==0)
                padding_mask = torch.ones(mask.shape).type_as(mask)
                for j in range(features.shape[-1]):
                    padding_mask *= (features[:,:,j]==0)
                mask *= ~padding_mask

            elif i == 9:
                prev_interval = 0.05*i - 0.005
                mask = (prev_interval < features[:,:,-1])

            else:
                prev_interval = 0.05*i - 0.005
                cur_interval = 0.05*i + 0.005
                mask = ((prev_interval <= features[:,:,-1]) & (features[:,:,-1] < cur_interval))

            split_features.append(self.temp_voxelize(features, mask))
        
        temp_features = self.point_motion_encoding(split_features)
        return temp_features.contiguous()
        
    def point_motion_encoding(self, split_features):
        cur_feats = split_features[0]
        mot_feats = []
        for i in range(1,self.nsweeps):
            mot_vec = self.layer1(cur_feats-split_features[i])
            mot_feats.append(self.layer2(mot_vec) * mot_vec)

        temp_features = self.layer3(torch.cat(mot_feats,1))

        return temp_features
    
    def temp_voxelize(self, features, mask):
        temp_features = features * mask.view(-1,self.nsweeps, 1)
        temp_num_voxels = mask.sum(1)
        temp_num_voxels[temp_num_voxels==0] = 1
        temp_points_mean = temp_features[:, :, : self.num_input_features].sum(
            dim=1, keepdim=False
        ) / temp_num_voxels.type_as(temp_features).view(-1, 1)

        return temp_points_mean