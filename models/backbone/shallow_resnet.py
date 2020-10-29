# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
from torch.utils import model_zoo
from models.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
}


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1, BatchNorm=None):
        super(BasicBlock, self).__init__()
        # self.conv1 = conv3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                     padding=dilation, dilation=dilation, bias=False)
        self.bn1 = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)
        # self.conv2 = conv3x3(planes, planes, stride=1, dilation=dilation)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                     padding=dilation, dilation=dilation, bias=False)
        self.bn2 = BatchNorm(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, backbone, block, layers, BatchNorm, in_channels=3, 
                 pretrained=True):
        self.backbone = backbone
        
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = BatchNorm(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], BatchNorm=BatchNorm)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, BatchNorm=BatchNorm)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2, BatchNorm=BatchNorm)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4, BatchNorm=BatchNorm)
        
        self._init_weight()

        if pretrained:
            self._load_pretrained_model()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample,  BatchNorm=BatchNorm)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, BatchNorm=BatchNorm))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        low_level_feat = x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x, low_level_feat
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def _load_pretrained_model(self):
        if self.backbone == 'resnet34':
            pretrain_dict = model_zoo.load_url(model_urls['resnet34'])
        elif self.backbone == 'resnet18':
            pretrain_dict = model_zoo.load_url(model_urls['resnet18'])
        else:
            NotImplementedError
                
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)


def resnet18(in_channels, BatchNorm, pretrained=True):
    if in_channels!=3 and pretrained:
        raise ValueError("pretraining is allowed only if input channels is 3")
    model = ResNet('resnet18', BasicBlock, [2, 2, 2, 2], BatchNorm, 
                   in_channels=in_channels, pretrained=pretrained)
    return model


def resnet34(in_channels, BatchNorm, pretrained=True):
    if in_channels!=3 and pretrained:
        raise ValueError("pretraining is allowed only if input channels is 3")
    model = ResNet('resnet34', BasicBlock, [3, 4, 6, 3], BatchNorm, 
                   in_channels=in_channels, pretrained=pretrained)
    return model

'''
import time
start = time.time()
if __name__ == "__main__":
    model = resnet18(in_channels=3, BatchNorm=nn.BatchNorm2d, pretrained=True)
    input = torch.rand(1, 3, 256, 256)
    output, low_level_feat = model(input)
    print(output.size())
    print(low_level_feat.size())
end = time.time()
print(end-start)
'''

