from models.backbone import resnet,shallow_resnet, xception, drn, mobilenet

def build_backbone(backbone, in_channels, output_stride, BatchNorm, pretrained):
    if backbone == 'resnet101':
        return resnet.resnet101(in_channels, output_stride, BatchNorm, pretrained)
    elif backbone == 'resnet50':
        return resnet.resnet50(in_channels, output_stride, BatchNorm, pretrained)
    elif backbone == 'resnet34':
        return shallow_resnet.resnet34(in_channels, BatchNorm, pretrained)
    elif backbone == 'resnet18':
        return shallow_resnet.resnet18(in_channels, BatchNorm, pretrained)
    elif backbone == 'xception':
        return xception.AlignedXception(output_stride, BatchNorm)
    elif backbone == 'drn':
        return drn.drn_d_54(BatchNorm)
    elif backbone == 'mobilenet':
        return mobilenet.MobileNetV2(output_stride, BatchNorm)
    else:
        raise NotImplementedError
