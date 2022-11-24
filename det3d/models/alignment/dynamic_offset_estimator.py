import torch
from torch import nn
from torch.nn import functional as F

from .non_local_block import NONLocalBlock2D


class Dynamic_Offset_Extimator(nn.Module):
    def __init__(self, in_channel_list, encode_out_channel, mot_mode):
        super(Dynamic_Offset_Extimator, self).__init__()
        
        encode_layer = []
        non_local_block = []
        for ms_idx, input_channel_size in enumerate(input_channel_list):
            if mot_mode == 'concat':
                encode_layer.append(nn.Sequential(
                    nn.Conv2d(in_channels=input_channel_size*2, out_channels=64,
                            kernel_size=3, stride=1, padding=1, bias=True),
                    nn.LeakyReLU(inplace=True)
                ))
            else:
                encode_layer.append(nn.Sequential(
                    nn.Conv2d(in_channels=input_channel_size, out_channels=64,
                            kernel_size=3, stride=1, padding=1, bias=True),
                    nn.LeakyReLU(inplace=True)
                ))
            non_local_block.append(NONLocalBlock2D(in_channels=64))

    def forward(self, x, return_nl_map=False):
        """
        :param x: (b, c, t, h, w)
        :param return_nl_map: if True return z, nl_map, else only return z.
        :return:
        """

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        if return_nl_map:
            return z, f_div_C
        return z