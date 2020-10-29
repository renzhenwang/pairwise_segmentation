# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d


class ConvBnRelu(nn.Module):
    def __init__(self, in_planes, out_planes, ksize, stride, pad, dilation=1,
                 groups=1, has_bn=True, norm_layer=nn.BatchNorm2d, bn_eps=1e-5,
                 has_relu=True, inplace=True, has_bias=False):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=ksize,
                              stride=stride, padding=pad,
                              dilation=dilation, groups=groups, bias=has_bias)
        self.has_bn = has_bn
        if self.has_bn:
            self.bn = norm_layer(out_planes, eps=bn_eps)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)

        return x


class SELayer(nn.Module):
    def __init__(self, in_planes, out_planes, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_planes, out_planes // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(out_planes // reduction, out_planes),
            nn.Sigmoid()
        )
        self.out_planes = out_planes

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, self.out_planes, 1, 1)
        return y


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, out_planes, reduction):
        super(ChannelAttention, self).__init__()
        self.channel_attention = SELayer(in_planes, out_planes, reduction)

    def forward(self, x1, x2):
        fm = torch.cat([x1, x2], 1)
        channel_weight = self.channel_attention(fm)
        fm = x1 * channel_weight + x2

        return fm


class ChannelFusion(nn.Module):
    def __init__(self, num_classes, backbone, BatchNorm, is_concat=False):
        super(ChannelFusion, self).__init__()
        if backbone in ['resnet50', 'resnet101', 'drn']:
            low_level_inplanes = 256
        elif backbone == 'resnet34' or backbone == 'resnet18':
            low_level_inplanes = 64
        elif backbone == 'xception':
            low_level_inplanes = 128
        elif backbone == 'mobilenet':
            low_level_inplanes = 24
        else:
            raise NotImplementedError
        
        self.concat = is_concat
        if self.concat:
            low_level_inplanes = 2*low_level_inplanes
            high_level_inplanes = 2*256
        else:
            low_level_inplanes = low_level_inplanes
            high_level_inplanes = 256

        self.conv_low_feat = ConvBnRelu(low_level_inplanes, 128, 3, 1, 1)
        self.conv_high_feat = ConvBnRelu(high_level_inplanes, 128, 3, 1, 1)
        
        self.channel_attention = ChannelAttention(256, 128, 1)
        
        self.last_conv = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
                               BatchNorm(64),
                               nn.ReLU(),
                               nn.Dropout(0.1),
                               nn.Conv2d(64, num_classes, kernel_size=1, stride=1))
        self._init_weight()


    def forward(self, br1_x, low_level_feat1, br2_x, low_level_feat2):
        if self.concat:
            fusion_low_feat = torch.cat((low_level_feat1, low_level_feat2), dim=1)
            fusion_high_feat = torch.cat((br1_x, br2_x), dim=1)
        else:
            fusion_low_feat = low_level_feat1 + low_level_feat2
            fusion_high_feat = br1_x + br2_x
        fusion_low_feat = self.conv_low_feat(fusion_low_feat)
        fusion_high_feat = self.conv_high_feat(fusion_high_feat)

        fusion_high_feat = F.interpolate(fusion_high_feat, size=low_level_feat2.size()[2:], mode='bilinear', align_corners=True)
        
        x = self.channel_attention(fusion_low_feat, fusion_high_feat)
        x = self.last_conv(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def build_attention_fusion(num_classes, backbone, BatchNorm, is_concat=False):
    return ChannelFusion(num_classes, backbone, BatchNorm, is_concat=is_concat)