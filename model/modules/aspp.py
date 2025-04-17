import torch
import torch.nn as nn
import torch.nn.functional as F

class _ASPPModule(nn.Module):
    def __init__(self, in_channels, planes, kernel_size, padding, dilation):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(in_channels, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)


class ASPP(nn.Module):
    def __init__(self, in_channels=64, out_channels=None, output_stride=16):
        super(ASPP, self).__init__()
        self.down_scale = 1
        if out_channels is None:
            out_channels = in_channels
        self.in_channelster = 256 // self.down_scale
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = _ASPPModule(in_channels, self.in_channelster, 1, padding=0, dilation=dilations[0])
        self.aspp2 = _ASPPModule(in_channels, self.in_channelster, 3, padding=dilations[1], dilation=dilations[1])
        self.aspp3 = _ASPPModule(in_channels, self.in_channelster, 3, padding=dilations[2], dilation=dilations[2])
        self.aspp4 = _ASPPModule(in_channels, self.in_channelster, 3, padding=dilations[3], dilation=dilations[3])

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(in_channels, self.in_channelster, 1, stride=1, bias=False),
                                             nn.BatchNorm2d(self.in_channelster),
                                             nn.ReLU(inplace=True))
        self.conv1 = nn.Conv2d(self.in_channelster * 5, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x1.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)
