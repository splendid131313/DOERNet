import torch.nn as nn


class LateralBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, inter_channels=64):
        super(LateralBlock, self).__init__()
        # inter_channels = in_channels // 4 if config.dec_channels_inter == 'adap' else 64
        self.conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        x = self.conv(x)
        return x


if __name__ == "__main__":
    block = LateralBlock()

    print(block)