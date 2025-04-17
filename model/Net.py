import torch
from kornia.filters import laplacian, gaussian_blur2d
from torch import nn
import torch.nn.functional as F

from backbone.pvt_v2 import pvt_v2_b2, pvt_v2_b4
from modules.decoder import DecoderBlock
from modules.lateral_blocks import LateralBlock
from modules.edge_prompt import EdgePrompt
from modules.orientation_enhance import OrientationEnhance


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class NeighborConnectionDecoder(nn.Module):
    def __init__(self, channel=64):
        super(NeighborConnectionDecoder, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv4 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3*channel, 1, 1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(x2_1)) * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        x = self.conv5(x)

        return x


class Encoder(nn.Module):
    def __init__(self, channel=64, channels=[1024, 640, 256, 128], pretrained=True):
        super(Encoder, self).__init__()

        self.backbone = pvt_v2_b4(pretrained=pretrained)  # [64, 128, 320, 512]

        # load model
        path = '../DOERNet/weights/pvt_v2_b4.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        self.conv_down1 = BasicConv2d(channels[3], channel, kernel_size=3, padding=1)  #
        self.conv_down2 = BasicConv2d(channels[2], channel, kernel_size=3, padding=1)
        self.conv_down3 = BasicConv2d(channels[1], channel, kernel_size=3, padding=1)
        self.conv_down4 = BasicConv2d(channels[0], channel, kernel_size=3, padding=1)

    def forward_enc(self, x):
        x1, x2, x3, x4 = self.backbone(x)
        B, C, H, W = x.shape
        x1_, x2_, x3_, x4_ = self.backbone(F.interpolate(x, size=(H//2, W//2), mode='bilinear', align_corners=True))

        x1 = torch.cat([x1, F.interpolate(x1_, size=x1.shape[2:], mode='bilinear', align_corners=True)], dim=1)
        x2 = torch.cat([x2, F.interpolate(x2_, size=x2.shape[2:], mode='bilinear', align_corners=True)], dim=1)
        x3 = torch.cat([x3, F.interpolate(x3_, size=x3.shape[2:], mode='bilinear', align_corners=True)], dim=1)
        x4 = torch.cat([x4, F.interpolate(x4_, size=x4.shape[2:], mode='bilinear', align_corners=True)], dim=1)

        x1 = self.conv_down1(x1)
        x2 = self.conv_down2(x2)
        x3 = self.conv_down3(x3)
        x4 = self.conv_down4(x4)

        return x1, x2, x3, x4

    def forward(self, x):
        x1, x2, x3, x4 = self.forward_enc(x)
        features = [x1, x2, x3, x4]
        features.append(laplacian(torch.mean(
            gaussian_blur2d(x, kernel_size=5, sigma=(1.0, 1.0)), dim=1).unsqueeze(1), kernel_size=5))

        return features


class Net(nn.Module):
    def __init__(self, channel=64, channels=[64, 128, 320, 512], img_size=[112, 56, 28, 14]):
        super(Net, self).__init__()

        self.encoder = Encoder(channel=channel)

        self.NCD = NeighborConnectionDecoder(channel)

        self.ep1 = EdgePrompt(img_size=img_size[0], out_channels=channels[0], depth=1)
        self.ep2 = EdgePrompt(img_size=img_size[1], out_channels=channels[1], depth=2)
        self.ep3 = EdgePrompt(img_size=img_size[2], out_channels=channels[2], depth=3)
        self.ep4 = EdgePrompt(img_size=img_size[3], out_channels=channels[3], depth=4)

        self.OEM1 = OrientationEnhance(in_channels=channel, out_channels=channel)
        self.OEM2 = OrientationEnhance(in_channels=channel, out_channels=channel)
        self.OEM3 = OrientationEnhance(in_channels=channel, out_channels=channel)
        self.OEM4 = OrientationEnhance(in_channels=channel, out_channels=channel)

        self.decoder4 = DecoderBlock(inter_channels=channel, out_channels=channel)
        self.decoder3 = DecoderBlock(inter_channels=channel, out_channels=channel)
        self.decoder2 = DecoderBlock(inter_channels=channel, out_channels=channel)
        self.decoder1 = DecoderBlock(inter_channels=channel, out_channels=channel)
        self.conv_out1 = nn.Sequential(nn.Conv2d(channel, 1, 1, 1, 0))

        self.lateral_block5 = LateralBlock(1, channel)
        self.lateral_block4 = LateralBlock(channel, channel)
        self.lateral_block3 = LateralBlock(channel, channel)
        self.lateral_block2 = LateralBlock(channel, channel)

        self.conv_ms_spvn_5 = nn.Conv2d(1, 1, 1, 1, 0)
        self.conv_ms_spvn_4 = nn.Conv2d(channel, 1, 1, 1, 0)
        self.conv_ms_spvn_3 = nn.Conv2d(channel, 1, 1, 1, 0)
        self.conv_ms_spvn_2 = nn.Conv2d(channel, 1, 1, 1, 0)

        _N = 16
        self.gdt_convs_4 = nn.Sequential(nn.Conv2d(channel, _N, 3, 1, 1), nn.BatchNorm2d(_N), nn.ReLU(inplace=True))
        self.gdt_convs_3 = nn.Sequential(nn.Conv2d(channel, _N, 3, 1, 1), nn.BatchNorm2d(_N), nn.ReLU(inplace=True))
        self.gdt_convs_2 = nn.Sequential(nn.Conv2d(channel, _N, 3, 1, 1), nn.BatchNorm2d(_N), nn.ReLU(inplace=True))

        self.gdt_convs_pred_5 = nn.Sequential(nn.Conv2d(1, 1, 1, 1, 0))
        self.gdt_convs_pred_4 = nn.Sequential(nn.Conv2d(_N, 1, 1, 1, 0))
        self.gdt_convs_pred_3 = nn.Sequential(nn.Conv2d(_N, 1, 1, 1, 0))
        self.gdt_convs_pred_2 = nn.Sequential(nn.Conv2d(_N, 1, 1, 1, 0))

        self.gdt_convs_attn_5 = nn.Sequential(nn.Conv2d(1, 1, 1, 1, 0))
        self.gdt_convs_attn_4 = nn.Sequential(nn.Conv2d(_N, 1, 1, 1, 0))
        self.gdt_convs_attn_3 = nn.Sequential(nn.Conv2d(_N, 1, 1, 1, 0))
        self.gdt_convs_attn_2 = nn.Sequential(nn.Conv2d(_N, 1, 1, 1, 0))

    def forward(self, x):
        outs_gdt_pred = []
        outs_gdt_label = []
        x1, x2, x3, x4, E_gt = self.encoder(x)

        outs = []

        # 5
        x5 = self.NCD(x4, x3, x2)
        p5 = F.interpolate(x5, scale_factor=8, mode='bilinear')
        m5 = self.conv_ms_spvn_5(p5)
        if self.training:
            # >> GT
            m5_dia = m5
            gdt_label_main_5 = E_gt * F.interpolate(m5_dia, size=E_gt.shape[2:], mode='bilinear', align_corners=True)
            outs_gdt_label.append(gdt_label_main_5)
            # >> Pred
            gdt_pred_5 = self.gdt_convs_pred_5(p5)
            outs_gdt_pred.append(gdt_pred_5)
        gdt_attn_5 = self.gdt_convs_attn_5(p5).sigmoid()
        p5 = p5 * gdt_attn_5

        # 4
        _p5 = F.interpolate(p5, size=x4.shape[2:], mode='bilinear', align_corners=True)
        x4 = x4 + self.lateral_block5(_p5)
        p4 = self.OEM4(x4, E_gt)
        edge_embedding4, image_pe4 = self.ep4(E_gt)
        p4 = self.decoder4(p4, image_pe4, edge_embedding4)
        m4 = self.conv_ms_spvn_4(p4)
        p4_gdt = self.gdt_convs_4(p4)
        if self.training:
            # >> gt:
            m4_dia = m4
            gdt_label_main_4 = E_gt * F.interpolate(m4_dia, size=E_gt.shape[2:], mode='bilinear', align_corners=True)
            outs_gdt_label.append(gdt_label_main_4)
            # >> Pred:
            gdt_pred_4 = self.gdt_convs_pred_4(p4_gdt)
            outs_gdt_pred.append(gdt_pred_4)
        gdt_attn_4 = self.gdt_convs_attn_4(p4_gdt).sigmoid()
        # >> Finally:
        p4 = p4 * gdt_attn_4

        # 3
        _p4 = F.interpolate(p4, size=x3.shape[2:], mode='bilinear', align_corners=True)
        _p3 = _p4 + self.lateral_block4(x3)
        p3 = self.OEM3(_p3, E_gt)
        edge_embedding3, image_pe3 = self.ep3(E_gt)
        p3 = self.decoder3(p3, image_pe3, edge_embedding3)
        m3 = self.conv_ms_spvn_3(p3)
        p3_gdt = self.gdt_convs_3(p3)
        if self.training:
            # >> GT:
            m3_dia = m3
            gdt_label_main_3 = E_gt * F.interpolate(m3_dia, size=E_gt.shape[2:], mode='bilinear', align_corners=True)
            outs_gdt_label.append(gdt_label_main_3)
            # >> Pred:
            gdt_pred_3 = self.gdt_convs_pred_3(p3_gdt)
            outs_gdt_pred.append(gdt_pred_3)
        gdt_attn_3 = self.gdt_convs_attn_3(p3_gdt).sigmoid()
        # >> Finally:
        p3 = p3 * gdt_attn_3

        # 2
        _p3 = F.interpolate(p3, size=x2.shape[2:], mode='bilinear', align_corners=True)
        _p2 = _p3 + self.lateral_block3(x2)
        p2 = self.OEM2(_p2, E_gt)
        edge_embedding2, image_pe2 = self.ep2(E_gt)
        p2 = self.decoder2(p2, image_pe2, edge_embedding2)
        m2 = self.conv_ms_spvn_2(p2)
        p2_gdt = self.gdt_convs_2(p2)
        if self.training:
            # >> GT:
            m2_dia = m2
            gdt_label_main_2 = E_gt * F.interpolate(m2_dia, size=E_gt.shape[2:], mode='bilinear', align_corners=True)
            outs_gdt_label.append(gdt_label_main_2)
            # >> Pred:
            gdt_pred_2 = self.gdt_convs_pred_2(p2_gdt)
            outs_gdt_pred.append(gdt_pred_2)
        gdt_attn_2 = self.gdt_convs_attn_2(p2_gdt).sigmoid()
        # >> Finally:
        p2 = p2 * gdt_attn_2

        # 1
        _p2 = F.interpolate(p2, size=x1.shape[2:], mode='bilinear', align_corners=True)
        _p1 = _p2 + self.lateral_block2(x1)
        _p1 = self.OEM1(_p1, E_gt)
        edge_embedding1, image_pe1 = self.ep1(E_gt)
        _p1 = self.decoder1(_p1, image_pe1, edge_embedding1)
        _p1 = F.interpolate(_p1, size=x.shape[2:], mode='bilinear', align_corners=True)
        p1_out = self.conv_out1(_p1)

        outs.append(m5)
        outs.append(m4)
        outs.append(m3)
        outs.append(m2)
        outs.append(p1_out)
        return outs if not self.training else ([outs_gdt_pred, outs_gdt_label], outs)
