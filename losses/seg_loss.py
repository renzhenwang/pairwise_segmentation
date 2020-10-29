import torch
import torch.nn as nn
import torch.nn.functional as F


class SegLoss(object):
    def __init__(self, weight=None, size_average=True, batch_average=True, ignore_index=255, cuda=False):
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda

    def build_loss(self, mode='ce'):
        """Choices: ['ce' or 'focal']"""
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        elif mode == 'dice':
            return self.BinaryDiceLoss
        elif mode == 'multi_dice':
            return self.MultitaskDiceLoss
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, logit, target):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= n

        return loss

    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss
    
    def BinaryDiceLoss(self, output, target):
        """ requires one hot encoded target. Applies DiceLoss on each class iteratively.
        requires input.shape[0:1] and target.shape[0:1] to be (N, C) where N is
        batch size and C is number of classes""" 
        if not isinstance(target, torch.FloatTensor):
            target = target.float()

        if output.dim()>2:
            output = output.permute(0, 2, 3, 1).contiguous()
            output = F.softmax(output, dim=-1)
            output = output[..., 1]

        output = output.view(-1)
        target = target.view(-1)

        smooth = 1.
        intersection = (output * target).sum() + smooth
        union = output.sum() + target.sum() + smooth

        return 1 - intersection / union

'''
if __name__ == "__main__":
    loss = SegLoss(cuda=True)
    a = torch.rand(1, 3, 7, 7).cuda()
    b = torch.rand(1, 7, 7).cuda()
    print(loss.MulticlassDiceLoss(a, b).item())
    print(loss.CrossEntropyLoss(a, b).item())
    print(loss.FocalLoss(a, b, gamma=0, alpha=None).item())
    print(loss.FocalLoss(a, b, gamma=2, alpha=0.5).item())
 '''