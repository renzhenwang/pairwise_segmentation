# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

class Fusion(nn.Module):
    def __init__(self, num_classes, backbone, BatchNorm, is_concat=False):
        super(Fusion, self).__init__()
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

        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = BatchNorm(48)
        self.relu = nn.ReLU()
        
        self.conv_low_feat = nn.Sequential(nn.Conv2d(low_level_inplanes, 96, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(96),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(96, 48, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(48),
                                       nn.ReLU())
        
        self.conv_high_feat = nn.Sequential(nn.Conv2d(high_level_inplanes, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU())
        
        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1, bias=False),
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

        x = F.interpolate(fusion_high_feat, size=low_level_feat2.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, fusion_low_feat), dim=1)
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

def build_fusion(num_classes, backbone, BatchNorm, is_concat=False):
    return Fusion(num_classes, backbone, BatchNorm, is_concat=is_concat)