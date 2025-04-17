import torch
from torch import nn
from rfb import RFBDeformable


class OrientationEnhance(nn.Module):
    def __init__(self, in_channels=64, out_channels=64):
        super(OrientationEnhance, self).__init__()
        inter_channels = in_channels // 4
        self.conv_in = nn.Conv2d(in_channels, inter_channels, 3, 1, padding=1)
        self.relu_in = nn.ReLU(inplace=True)
        self.dec_att = RFBDeformable(in_channels=inter_channels)
        self.conv_out = nn.Conv2d(inter_channels, out_channels, 3, 1, padding=1)
        self.bn_in = nn.BatchNorm2d(inter_channels)
        self.bn_out = nn.BatchNorm2d(out_channels)

    def forward(self, x, gdt_gt):
        x = self.conv_in(x)
        x = self.bn_in(x)
        x = self.relu_in(x)
        x = self.dec_att(x, gdt_gt)
        x = self.conv_out(x)
        x = self.bn_out(x)
        return x