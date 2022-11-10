import torch
import torch.nn as nn

from nets.resnet import resnet50
from nets.vgg import VGG16

from torch.nn import functional as F

class SoftPooling2D(torch.nn.Module):
    def __init__(self,kernel_size,strides=None,padding=0,ceil_mode = False,count_include_pad = True,divisor_override = None):
        super(SoftPooling2D, self).__init__()
        self.avgpool = torch.nn.AvgPool2d(kernel_size,strides,padding,ceil_mode,count_include_pad,divisor_override)
    def forward(self, x):
        x_exp = torch.exp(x)
        x_exp_pool = self.avgpool(x_exp)
        x = self.avgpool(x_exp*x)
        return x/x_exp_pool

class SEAttention(nn.Module):
    def __init__(self, channel=512, reduction=16):
        super().__init__()
        # 池化层，将每一个通道的宽和高都变为 1 (平均值)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            # 先降低维
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            # 再升维
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        # y是权重
        return x * y.expand_as(x)




class depthwise_conv(nn.Module):
    def __init__(self, ch_in, kernel_size):
        super(depthwise_conv, self).__init__()
        self.ch_in = ch_in
        self.kernel_size = kernel_size
        # self.ch_out = ch_out
        self.depth_conv = nn.Conv2d(ch_in, ch_in, kernel_size=kernel_size, padding=int((kernel_size-1)/2), groups=ch_in)



    def forward(self, x):
        x = self.depth_conv(x)
        return x

class MSDM(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(MSDM, self).__init__()
        self.DSC = depthwise_conv(in_channel, 3)
        self.DSC = depthwise_conv(in_channel, 5)
        self.DSC = depthwise_conv(in_channel, 7)
        self.bn = nn.BatchNorm2d(in_channel)
        self.conv = nn.Conv2d(3*in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.SE = SEAttention(channel = out_channel, reduction = 8)
        self.SELU = nn.SELU()

    def forward(self, x):
        x1 = self.DSC(x)
        x2 = self.DSC(x)
        x3 = self.DSC(x)
        x1 = self.bn(x1)
        x2 = self.bn(x2)
        x3 = self.bn(x3)
        x1 = self.SELU(x1)
        x2 = self.SELU(x2)
        x3 = self.SELU(x3)
        x = torch.cat([x1, x2, x3], 1)
        x = self.conv(x)
        x = self.SELU(x)
        x = self.SE(x)

        return x





class AttentionGateBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGateBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g = self.W_g(g)
        x = self.W_x(x)
        psi = self.relu(g + x)
        psi = self.psi(psi)

        return x * psi


class AttentionGateOut(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGateOut, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g = self.W_g(g)
        x = self.W_x(x)
        psi = self.relu(g + x)
        psi = self.psi(psi)

        return x * psi


class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, inputs2], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs


class Unet(nn.Module):
    def __init__(self, num_classes=21, pretrained=False, backbone='vgg'):
        super(Unet, self).__init__()

        self.MSDM1 = MSDM(3, 64)
        self.MSDM2 = MSDM(64, 128)
        self.MSDM3 = MSDM(128, 256)
        self.MSDM4 = MSDM(256, 512)
        self.MSDM5 = MSDM(512, 512)
        self.softpool = SoftPooling2D(kernel_size=2)
        in_filters = [128, 256, 512, 1024]



        # if backbone == 'vgg':
        #     self.vgg = VGG16(pretrained=pretrained)
        #     in_filters = [128, 256, 512, 1024]
        # elif backbone == "resnet50":
        #     self.resnet = resnet50(pretrained=pretrained)
        #     in_filters = [192, 512, 1024, 3072]
        # else:
        #     raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet50.'.format(backbone))
        out_filters = [32, 64, 128, 256, 960]

        # AG_Gate in skip_connection
        self.Attentiongate1 = AttentionGateBlock(512, 512, 512)
        self.Attentiongate2 = AttentionGateBlock(256, 256, 256)
        self.Attentiongate3 = AttentionGateBlock(128, 128, 128)
        self.Attentiongate4 = AttentionGateBlock(64, 64, 64)

        # AG_Gate in decode_connection
        self.Attentiongateout1 = AttentionGateOut(256, 512, 512)
        self.Attentiongateout2 = AttentionGateOut(128, 256, 256)
        self.Attentiongateout3 = AttentionGateOut(64, 128, 128)
        self.Attentiongateout4 = AttentionGateOut(32, 64, 64)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        # upsampling
        # 64,64,512
        self.up_concat4 = unetUp(in_filters[3], out_filters[3])
        # 128,128,256
        self.up_concat3 = unetUp(in_filters[2], out_filters[2])
        # 256,256,128
        self.up_concat2 = unetUp(in_filters[1], out_filters[1])
        # 512,512,64
        self.up_concat1 = unetUp(in_filters[0], out_filters[0])

        if backbone == 'resnet50':
            self.up_conv = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
                nn.ReLU(),
            )
        else:
            self.up_conv = None

        self.final = nn.Conv2d(out_filters[4], num_classes, 1)

        self.backbone = backbone

    def forward(self, inputs):
        # if self.backbone == "vgg":
        #     [feat1, feat2, feat3, feat4, feat5] = self.vgg.forward(inputs)
        # elif self.backbone == "resnet50":
        #     [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(inputs)

        out_h = 256
        out_w = 256

        feat1 = self.MSDM1(inputs)
        feat2 = self.softpool(feat1)
        feat2 = self.MSDM2(feat2)
        feat3 = self.softpool(feat2)
        feat3 = self.MSDM3(feat3)
        feat4 = self.softpool(feat3)
        feat4 = self.MSDM4(feat4)
        feat5 = self.softpool(feat4)
        feat5 = self.MSDM5(feat5)

        feat5 = self.up(feat5)
        feat4 = self.Attentiongate1(feat4, feat5)
        up4 = self.up_concat4(feat4, feat5)
        new_up4 = self.Attentiongateout1(up4, feat5)
        AG_out4 = F.interpolate(new_up4, scale_factor=8, mode='bilinear', align_corners=True)


        up4 = self.up(up4)
        feat3 = self.Attentiongate2(feat3, up4)
        up3 = self.up_concat3(feat3, up4)
        new_up3 = self.Attentiongateout2(up3, up4)
        AG_out3 = F.interpolate(new_up3, scale_factor=4, mode='bilinear', align_corners=True)

        up3 = self.up(up3)
        feat2 = self.Attentiongate3(feat2, up3)
        up2 = self.up_concat2(feat2, up3)
        new_up2 = self.Attentiongateout3(up2, up3)
        AG_out2 = F.interpolate(new_up2, scale_factor=2, mode='bilinear', align_corners=True)

        up2 = self.up(up2)
        feat1 = self.Attentiongate4(feat1, up2)
        up1 = self.up_concat1(feat1, up2)
        AG_out1 = self.Attentiongateout4(up1, up2)

        out = torch.cat([AG_out1, AG_out2, AG_out3, AG_out4], 1)
        if self.up_conv != None:
            up1 = self.up_conv(up1)

        final = self.final(out)

        return final

    def freeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.MSDM1.parameters():
                param.requires_grad = False
            for param in self.MSDM2.parameters():
                param.requires_grad = False
            for param in self.MSDM3.parameters():
                param.requires_grad = False
            for param in self.MSDM4.parameters():
                param.requires_grad = False
            for param in self.MSDM5.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.MSDM1.parameters():
                param.requires_grad = True
            for param in self.MSDM2.parameters():
                param.requires_grad = True
            for param in self.MSDM3.parameters():
                param.requires_grad = True
            for param in self.MSDM4.parameters():
                param.requires_grad = True
            for param in self.MSDM5.parameters():
                param.requires_grad = True