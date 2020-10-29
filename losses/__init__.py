# -*- coding: utf-8 -*-

from __future__ import absolute_import

from .pairwise_loss import MultiClassDiceLoss, MultiLabelBCELoss, MultiLabelDiceLoss, PairwiseCompose
from .pairwise_loss import CELoss, DiceLoss

"""Create loss"""
__factory = {
    'ce': CELoss,
    'dice': DiceLoss,
    'multi_class_dice': MultiClassDiceLoss,
    'multi_label_dice': MultiLabelDiceLoss,
    'multi_label_ce': MultiLabelBCELoss,
    'pairwise_loss': PairwiseCompose,
    }


def get_names():
    return __factory.keys()


def init_loss(name, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown loss: {}".format(name))
    return __factory[name](**kwargs)
