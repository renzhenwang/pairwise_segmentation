# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from models.aspp import build_aspp
from models.decoder import build_decoder
from models.backbone import build_backbone
from models.fusion import build_fusion
from models.attention_fusion import build_attention_fusion

class PairwiseDeepLab(nn.Module):
    def __init__(self, backbone='resnet18', in_channels=3, output_stride=16, 
                 num_classes=1, aux_classes=3, sync_bn=True, freeze_bn=False, 
                 pretrained=False, fusion_type='fusion', is_concat=False,  **kwargs):
        super(PairwiseDeepLab, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, in_channels, output_stride, BatchNorm, pretrained)
        
        ## branch1
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)
        
        ## branch2
        # self.br2_aspp = build_aspp(backbone, output_stride, BatchNorm)
        # self.br2_decoder = build_decoder(num_classes, backbone, BatchNorm)
        
        ## fusion
        self.fusion_type = fusion_type
        if self.fusion_type == 'attention_fusion':
            print('fusion_type is attention_fusion')
            self.fusion = build_attention_fusion(aux_classes, backbone, BatchNorm, is_concat=is_concat)
        elif self.fusion_type == 'fusion':
            print('init fusion_type')
            self.fusion = build_fusion(aux_classes, backbone, BatchNorm, is_concat=is_concat)
        else:
            raise NotImplementedError
        
        if freeze_bn:
            self.freeze_bn()

    def forward(self, x1, x2):
        ## branch1
        br1_x, low_level_feat1 = self.backbone(x1)
        # print(br1_x.shape, low_level_feat1.shape)
        br1_x = self.aspp(br1_x)
        br1_out = self.decoder(br1_x, low_level_feat1)
        br1_out = F.interpolate(br1_out, size=x1.size()[2:], mode='bilinear', align_corners=True)
        # br1_out = br1_out.permute(0, 2, 3, 1).contiguous()
        
        ## branch2
        br2_x, low_level_feat2 = self.backbone(x2)
        br2_x = self.aspp(br2_x)
        br2_out = self.decoder(br2_x, low_level_feat2)
        br2_out = F.interpolate(br2_out, size=x2.size()[2:], mode='bilinear', align_corners=True)
        # br2_out = br2_out.permute(0, 2, 3, 1).contiguous()
        
        ## fusion
        fusion_x = self.fusion(br1_x, low_level_feat1, br2_x, low_level_feat2)
        fusion_x = F.interpolate(fusion_x, size=x2.size()[2:], mode='bilinear', align_corners=True)
        # fusion_x = fusion_x.permute(0, 2, 3, 1).contiguous()

        return br1_out, br2_out, fusion_x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder, self.fusion]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p


import time
start = time.time()
if __name__ == "__main__":
    model = PairwiseDeepLab(backbone='resnet18', output_stride=16, in_channels=5, 
                            pretrained=False, fusion_type='attention_fusion')
    model.eval()
    input = torch.rand(1, 5, 256, 256)
    output = model(input, input[:])
    print(output[0].size(), output[1].size(), output[2].size())
    print("Total paramerters: {}".format(sum(x.numel() for x in model.parameters())))   
end = time.time()
print(end-start)


